#!/usr/bin/env python3
"""Independent, third-party-free Decimal oracle for fixed-K FIEGARCH QML.

The contract is a one-AR/no-news-MA Gaussian-QML FIEGARCH recurrence.  The
fractional filter is deliberately finite and its runtime length ``K`` is part
of every fixture.  This is an oracle, not a production estimator: a small
deterministic grid supplies one seed per explicitly enumerated ``d`` face and
Decimal BFGS performs exactly one local refinement on each face.

Commands::

    python scripts/fiegarch_qml_oracle.py emit golden/fiegarch_qml.json
    python scripts/fiegarch_qml_oracle.py fast-check golden/fiegarch_qml.json
    python scripts/fiegarch_qml_oracle.py deep-check golden/fiegarch_qml.json

``deep-check`` rebuilds the raw tree at 80 and 120 digits and compares it
before output rounding.  It also audits analytic derivatives, an independent
direct recurrence, the d=0 EGARCH identity, sign and scale invariants,
truncation behavior, and ordered +infinity barriers.  Binary64 replay is
reported only to stdout because it is host dependent.

Reference: Bollerslev and Mikkelsen (1996), Eq. 11.  The infinite fractional
operator in that paper is represented here by the explicitly stated finite-K
truncation and fixed, parameter-independent initialization below.
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
from typing import Sequence


ORACLE_PRECISION = 80
VERIFICATION_PRECISION = 120
OUTPUT_SIGNIFICANT_DIGITS = 24
AGREEMENT_SIGNIFICANT_DIGITS = 24
NEAR_ZERO_ABSOLUTE_FLOOR = Decimal("1e-18")
NEAR_ZERO_ABSOLUTE_TOLERANCE = Decimal("1e-42")
GRADIENT_AUDIT_TOLERANCE = Decimal("3e-15")
KKT_TOLERANCE = Decimal("2e-20")
OPTIMIZER_TOLERANCE = Decimal("2e-24")
OPEN_BOUNDARY_GUARD = Decimal("1e-13")
MAX_BFGS_ITERATIONS = 320
MAX_LINE_SEARCH_STEPS = 120

# Replaced after the first independently regenerated artifact is reviewed.
EXPECTED_ARTIFACT_SHA256 = (
    "475a3329fb154f80bf3765d1b9ef9e9ae17bee8a728d68aef55998c58381c6b8"
)

SQRT_2_OVER_PI_TEXT = (
    "0.79788456080286535587989211986876373695171726232986931533185165934131585179860367700250466781461387286060511772527036537102198390911167448598"
)

ZERO = Decimal(0)
ONE = Decimal(1)
TWO = Decimal(2)
HALF = Decimal("0.5")


@dataclass(frozen=True)
class CaseSpec:
    name: str
    purpose: str
    observations: tuple[str, ...]
    truncation: int
    expected_selection: str
    expected_property: str
    reflection_of: str | None = None
    probe_parameters: tuple[str, str, str, str, str] | None = None


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
    physical_gradient: tuple[Decimal, Decimal, Decimal, Decimal, Decimal]
    log_variances: tuple[Decimal, ...]
    physical_variances: tuple[Decimal, ...]
    fractional_terms: tuple[Decimal, ...]
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
    d: Decimal
    evaluation: Evaluation
    transformed_gradient: tuple[Decimal, ...]
    iterations: int
    converged: bool
    open_boundary_guard_rejected: bool
    d_zero_directional_derivative: Decimal | None


@dataclass(frozen=True)
class SolveResult:
    selected: Candidate
    face_candidates: tuple[Candidate, ...]
    seed_grid: dict[str, object]


def _decimal_pi() -> Decimal:
    a = ONE
    b = ONE / TWO.sqrt()
    t = ONE / Decimal(4)
    p = ONE
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
    from decimal import getcontext

    checked_digits = min(getcontext().prec - 8, 120)
    if abs(calculated - embedded) > Decimal(10) ** Decimal(-checked_digits):
        raise AssertionError("embedded sqrt(2/pi) failed Decimal audit")
    return calculated


def _lcg_normal_surrogates(count: int, seed: int) -> tuple[Decimal, ...]:
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


def _fractional_coefficients(d: Decimal, truncation: int) -> tuple[Decimal, ...]:
    if truncation < 1:
        raise ValueError("truncation must be positive")
    coefficients = [ONE]
    for k in range(1, truncation):
        coefficients.append(coefficients[-1] * (Decimal(k - 1) + d) / Decimal(k))
    return tuple(coefficients)


def _fractional_coefficients_with_derivative(
    d: Decimal, truncation: int,
) -> tuple[tuple[Decimal, ...], tuple[Decimal, ...]]:
    coefficients = [ONE]
    derivatives = [ZERO]
    for k in range(1, truncation):
        ratio = (Decimal(k - 1) + d) / Decimal(k)
        coefficients.append(coefficients[-1] * ratio)
        derivatives.append(derivatives[-1] * ratio + coefficients[-2] / Decimal(k))
    return tuple(coefficients), tuple(derivatives)


def _simulate_fiegarch_observations(
    mu: str,
    alpha: str,
    gamma: str,
    beta: str,
    d: str,
    truncation: int,
    innovations: Sequence[Decimal],
) -> tuple[str, ...]:
    """Freeze deterministic fixtures; neither oracle evaluator calls this."""
    with localcontext() as ctx:
        ctx.prec = 70
        m, a, g, b, frac = map(Decimal, (mu, alpha, gamma, beta, d))
        coefficients = _fractional_coefficients(frac, truncation)
        c = Decimal(SQRT_2_OVER_PI_TEXT)
        logh = m
        news: list[Decimal] = []
        residuals: list[Decimal] = []
        for z in innovations:
            residuals.append((logh / TWO).exp() * z)
            news.append(a * (abs(z) - c) + g * z)
            upto = min(len(news), truncation)
            filtered = sum(
                (coefficients[k] * news[-1 - k] for k in range(upto)), ZERO
            )
            logh = (ONE - b) * m + b * logh + filtered
        mean = sum(residuals, ZERO) / Decimal(len(residuals))
        return tuple(format(value - mean, ".28g") for value in residuals)


def _negate(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(format(-Decimal(value), "f") for value in values)


_LONG_MEMORY = _simulate_fiegarch_observations(
    "-0.30", "0.20", "-0.22", "0.62", "0.28", 18,
    _lcg_normal_surrogates(90, 0xF1E6_1101),
)
_D_ZERO = _simulate_fiegarch_observations(
    "-0.22", "0.25", "-0.16", "0.55", "0", 18,
    _lcg_normal_surrogates(100, 0xD000_4047),
)
_NEGATIVE_BETA = _simulate_fiegarch_observations(
    "-0.18", "0.24", "0.12", "-0.48", "0.22", 16,
    _lcg_normal_surrogates(90, 0xBE7A_9091),
)
_NEAR_HALF = _simulate_fiegarch_observations(
    "-0.38", "0.08", "-0.08", "0.30", "0.465", 20,
    _lcg_normal_surrogates(90, 0xD499_0501),
)
_QUIET_EXTREME = ("1", "-1") + ("0",) * 398 + ("1", "-1")


CASES = (
    CaseSpec(
        "interior_long_memory",
        "Interior finite-K FIEGARCH fit with material fractional memory.",
        _LONG_MEMORY, 18, "interior", "d_material",
    ),
    CaseSpec(
        "interior_long_memory_reflected",
        "Exact sign reflection: gamma changes sign and variance quantities agree.",
        _negate(_LONG_MEMORY), 18, "interior", "reflection",
        reflection_of="interior_long_memory",
    ),
    CaseSpec(
        "d_zero_boundary",
        "Exact d=0 face, including its one-sided KKT derivative and EGARCH identity.",
        _D_ZERO, 18, "d_zero", "d_zero_face",
    ),
    CaseSpec(
        "negative_beta",
        "Negative-AR fit demonstrating the full beta domain; this fixture selects the exact d=0 face.",
        _NEGATIVE_BETA, 16, "d_zero", "beta_negative",
    ),
    CaseSpec(
        "near_d_half",
        "Finite valid recurrence near, but separated from, the open d=0.5 boundary.",
        _NEAR_HALF, 20, "probe", "d_near_half",
        probe_parameters=("-0.38", "0.08", "-0.08", "0.30", "0.465"),
    ),
    CaseSpec(
        "quiet_extreme_probe",
        "Valid Decimal recurrence remains ordered while binary64 QML reaches +infinity.",
        _QUIET_EXTREME, 16, "probe", "extended_real",
        probe_parameters=("-800", "0.10", "-0.10", "0.80", "0.30"),
    ),
    CaseSpec(
        "constant_failure",
        "Demeaning produces zero scale, so volatility parameters are unidentified.",
        tuple("7.125" for _ in range(24)), 12, "unidentified", "constant_failure",
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


def _sigmoid(value: Decimal) -> Decimal:
    if value >= ZERO:
        exp_negative = (-value).exp()
        return ONE / (ONE + exp_negative)
    exp_positive = value.exp()
    return exp_positive / (ONE + exp_positive)


def _decode(
    selection: str, coordinates: Sequence[Decimal],
) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]:
    if selection == "interior":
        mu, alpha, gamma, beta_coordinate, d_coordinate = coordinates
        beta = TWO * _sigmoid(TWO * beta_coordinate) - ONE
        d = HALF * _sigmoid(d_coordinate)
    elif selection == "d_zero":
        mu, alpha, gamma, beta_coordinate = coordinates
        beta = TWO * _sigmoid(TWO * beta_coordinate) - ONE
        d = ZERO
    else:
        raise ValueError(f"unknown face {selection!r}")
    omega = (ONE - beta) * mu
    return mu, omega, alpha, gamma, beta, d


def _objective_and_path_direct(
    prepared: PreparedSeries,
    mu: Decimal,
    alpha: Decimal,
    gamma: Decimal,
    beta: Decimal,
    d: Decimal,
    truncation: int,
) -> tuple[Decimal, tuple[Decimal, ...], tuple[Decimal, ...], Decimal]:
    """Independent direct recurrence: no derivative state or shared filter helper."""
    if (
        prepared.initial_log_variance is None
        or not (-ONE < beta < ONE)
        or not (ZERO <= d < HALF)
        or truncation < 1
    ):
        return Decimal("Infinity"), tuple(), tuple(), Decimal("NaN")
    try:
        coefficients = [ONE]
        for k in range(1, truncation):
            coefficients.append(
                coefficients[k - 1] * (Decimal(k - 1) + d) / Decimal(k)
            )
        c = Decimal(SQRT_2_OVER_PI_TEXT)
        logh = prepared.initial_log_variance
        objective = ZERO
        path: list[Decimal] = []
        fractions: list[Decimal] = []
        news: list[Decimal] = []
        for x in prepared.normalized:
            standardized_square = x * x * (-logh).exp()
            objective += HALF * (logh + standardized_square)
            path.append(logh)
            z = x * (-logh / TWO).exp()
            news.append(alpha * (abs(z) - c) + gamma * z)
            upto = min(len(news), truncation)
            filtered = ZERO
            for k in range(upto):
                filtered += coefficients[k] * news[len(news) - 1 - k]
            fractions.append(filtered)
            logh = (ONE - beta) * mu + beta * logh + filtered
        return objective, tuple(path), tuple(fractions), logh
    except (ArithmeticError, InvalidOperation):
        return Decimal("Infinity"), tuple(), tuple(), Decimal("NaN")


def _objective_only(
    prepared: PreparedSeries,
    mu: Decimal,
    alpha: Decimal,
    gamma: Decimal,
    beta: Decimal,
    d: Decimal,
    truncation: int,
) -> Decimal:
    return _objective_and_path_direct(
        prepared, mu, alpha, gamma, beta, d, truncation
    )[0]


def _extended_exp(value: Decimal) -> Decimal:
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
    d: Decimal,
    truncation: int,
) -> Evaluation:
    """Analytic recurrence and gradient in (omega, alpha, gamma, beta, d)."""
    if prepared.initial_log_variance is None:
        raise ValueError("constant series")
    if not (-ONE < beta < ONE) or not (ZERO <= d < HALF):
        raise ValueError("parameter outside FIEGARCH domain")
    coefficients, coefficient_d = _fractional_coefficients_with_derivative(d, truncation)
    c = Decimal(SQRT_2_OVER_PI_TEXT)
    omega = (ONE - beta) * mu
    logh = prepared.initial_log_variance
    derivative = [ZERO] * 5
    gradient = [ZERO] * 5
    objective = ZERO
    news: list[Decimal] = []
    news_derivatives: list[list[Decimal]] = []
    log_path: list[Decimal] = []
    variance_path: list[Decimal] = []
    fractional_path: list[Decimal] = []
    scale_square = prepared.scale * prepared.scale

    for x in prepared.normalized:
        z = x * (-logh / TWO).exp()
        standardized_square = z * z
        objective += HALF * (logh + standardized_square)
        score = HALF * (ONE - standardized_square)
        for j in range(5):
            gradient[j] += score * derivative[j]
        log_path.append(logh)
        variance_path.append(scale_square * _extended_exp(logh))

        q = abs(z) - c
        current_news = alpha * q + gamma * z
        current_news_derivative: list[Decimal] = []
        feedback = -HALF * (alpha * abs(z) + gamma * z)
        for j in range(5):
            direct = q if j == 1 else z if j == 2 else ZERO
            current_news_derivative.append(direct + feedback * derivative[j])
        news.append(current_news)
        news_derivatives.append(current_news_derivative)

        upto = min(len(news), truncation)
        filtered = ZERO
        filtered_derivative = [ZERO] * 5
        for k in range(upto):
            index = len(news) - 1 - k
            filtered += coefficients[k] * news[index]
            for j in range(5):
                filtered_derivative[j] += coefficients[k] * news_derivatives[index][j]
            filtered_derivative[4] += coefficient_d[k] * news[index]
        fractional_path.append(filtered)

        next_derivative = [ZERO] * 5
        for j in range(5):
            direct = ONE if j == 0 else logh if j == 3 else ZERO
            next_derivative[j] = direct + beta * derivative[j] + filtered_derivative[j]
        logh = omega + beta * logh + filtered
        derivative = next_derivative

    return Evaluation(
        objective,
        tuple(gradient),  # type: ignore[arg-type]
        tuple(log_path),
        tuple(variance_path),
        tuple(fractional_path),
        logh,
        scale_square * _extended_exp(logh),
    )


def _transformed_evaluation(
    prepared: PreparedSeries,
    selection: str,
    coordinates: Sequence[Decimal],
    truncation: int,
) -> tuple[Evaluation, tuple[Decimal, ...]]:
    mu, _, alpha, gamma, beta, d = _decode(selection, coordinates)
    evaluation = _evaluate(prepared, mu, alpha, gamma, beta, d, truncation)
    g_omega, g_alpha, g_gamma, g_beta, g_d = evaluation.physical_gradient
    g_mu = (ONE - beta) * g_omega
    beta_coordinate_gradient = (ONE - beta * beta) * (g_beta - mu * g_omega)
    if selection == "d_zero":
        return evaluation, (g_mu, g_alpha, g_gamma, beta_coordinate_gradient)
    d_coordinate_gradient = d * (ONE - TWO * d) * g_d
    return evaluation, (
        g_mu, g_alpha, g_gamma, beta_coordinate_gradient, d_coordinate_gradient,
    )


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
    scale = max(ONE, *(abs(v) for v in step), *(abs(v) for v in gradient_delta))
    if curvature <= Decimal("1e-62") * scale * scale:
        return _identity(size)
    rho = ONE / curvature
    left = [[
        (ONE if r == c else ZERO) - rho * step[r] * gradient_delta[c]
        for c in range(size)
    ] for r in range(size)]
    right = [[
        (ONE if r == c else ZERO) - rho * gradient_delta[r] * step[c]
        for c in range(size)
    ] for r in range(size)]
    middle = [[sum(
        (left[r][k] * inverse_hessian[k][c] for k in range(size)), ZERO
    ) for c in range(size)] for r in range(size)]
    return [[sum(
        (middle[r][k] * right[k][c] for k in range(size)), ZERO
    ) + rho * step[r] * step[c] for c in range(size)] for r in range(size)]


def _atanh(value: Decimal) -> Decimal:
    return ((ONE + value) / (ONE - value)).ln() / TWO


def _d_logit(value: Decimal) -> Decimal:
    probability = TWO * value
    return (probability / (ONE - probability)).ln()


def _initial_coordinates(
    selection: str,
    mu: Decimal,
    alpha: Decimal,
    gamma: Decimal,
    beta: Decimal,
    d: Decimal,
) -> tuple[Decimal, ...]:
    if selection == "interior":
        return (mu, alpha, gamma, _atanh(beta), _d_logit(d))
    return (mu, alpha, gamma, _atanh(beta))


def _open_boundary_guarded(selection: str, beta: Decimal, d: Decimal) -> bool:
    """Reject every open boundary, including interior saturation toward d=0."""
    slacks = [ONE - abs(beta)]
    if selection == "interior":
        slacks.extend((d, HALF - d))
    return min(slacks) <= OPEN_BOUNDARY_GUARD


def _bfgs(
    prepared: PreparedSeries,
    selection: str,
    initial: Sequence[Decimal],
    truncation: int,
) -> Candidate:
    coordinates = list(initial)
    evaluation, gradient_tuple = _transformed_evaluation(
        prepared, selection, coordinates, truncation
    )
    gradient = list(gradient_tuple)
    inverse_hessian = _identity(len(coordinates))
    converged = False
    iterations = 0

    for iteration in range(MAX_BFGS_ITERATIONS):
        iterations = iteration
        mu, omega, alpha, gamma, beta, d = _decode(selection, coordinates)
        del omega, alpha, gamma
        guarded = _open_boundary_guarded(selection, beta, d)
        if guarded:
            break
        if max(abs(value) for value in gradient) <= OPTIMIZER_TOLERANCE:
            converged = True
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
            trial = [old + step_size * delta for old, delta in zip(coordinates, direction)]
            try:
                next_evaluation, next_gradient_tuple = _transformed_evaluation(
                    prepared, selection, trial, truncation
                )
            except (ArithmeticError, InvalidOperation, ValueError):
                step_size *= HALF
                continue
            if next_evaluation.objective <= (
                evaluation.objective + Decimal("1e-4") * step_size * directional
            ):
                accepted = True
                break
            step_size *= HALF
        if not accepted:
            break
        next_coordinates = trial
        next_gradient = list(next_gradient_tuple)
        actual_step = [new - old for new, old in zip(next_coordinates, coordinates)]
        gradient_delta = [new - old for new, old in zip(next_gradient, gradient)]
        inverse_hessian = _inverse_bfgs_update(
            inverse_hessian, actual_step, gradient_delta
        )
        coordinates = next_coordinates
        evaluation = next_evaluation
        gradient = next_gradient
    else:
        iterations = MAX_BFGS_ITERATIONS

    mu, omega, alpha, gamma, beta, d = _decode(selection, coordinates)
    guarded = _open_boundary_guarded(selection, beta, d)
    d_direction = evaluation.physical_gradient[4] if selection == "d_zero" else None
    gradient_small = max(abs(value) for value in gradient) <= KKT_TOLERANCE
    kkt = d_direction is None or d_direction >= -KKT_TOLERANCE
    return Candidate(
        selection, tuple(coordinates), mu, omega, alpha, gamma, beta, d,
        evaluation, tuple(gradient), iterations,
        (converged or gradient_small) and not guarded and kkt,
        guarded, d_direction,
    )


def _insert_best(
    entries: list[tuple[Decimal, tuple[Decimal, Decimal, Decimal, Decimal, Decimal]]],
    item: tuple[Decimal, tuple[Decimal, Decimal, Decimal, Decimal, Decimal]],
) -> None:
    entries.append(item)
    entries.sort(key=lambda entry: entry[0])
    del entries[3:]


def _eligible(candidate: Candidate) -> bool:
    if not candidate.converged:
        return False
    if max(abs(value) for value in candidate.transformed_gradient) > KKT_TOLERANCE:
        return False
    return (
        candidate.d_zero_directional_derivative is None
        or candidate.d_zero_directional_derivative >= -KKT_TOLERANCE
    )


def _solve(prepared: PreparedSeries, truncation: int) -> SolveResult:
    if prepared.initial_log_variance is None:
        raise ValueError("constant series")
    mu_offsets = tuple(map(Decimal, ("-2", "0", "2")))
    alpha_values = tuple(map(Decimal, ("0", "0.20", "0.45")))
    gamma_values = tuple(map(Decimal, ("-0.30", "0", "0.30")))
    beta_values = tuple(map(Decimal, ("-0.70", "-0.20", "0.35", "0.75")))
    d_values = tuple(map(Decimal, ("0.04", "0.18", "0.34", "0.46")))
    best: dict[str, list[tuple[Decimal, tuple[Decimal, Decimal, Decimal, Decimal, Decimal]]]] = {
        "interior": [], "d_zero": [],
    }
    evaluations = 0
    for offset in mu_offsets:
        mu = prepared.initial_log_variance + offset
        for alpha in alpha_values:
            for gamma in gamma_values:
                for beta in beta_values:
                    objective = _objective_only(
                        prepared, mu, alpha, gamma, beta, ZERO, truncation
                    )
                    _insert_best(best["d_zero"], (
                        objective, (mu, alpha, gamma, beta, ZERO),
                    ))
                    evaluations += 1
                    for d in d_values:
                        objective = _objective_only(
                            prepared, mu, alpha, gamma, beta, d, truncation
                        )
                        _insert_best(best["interior"], (
                            objective, (mu, alpha, gamma, beta, d),
                        ))
                        evaluations += 1

    candidates: list[Candidate] = []
    refinement_objectives: dict[str, tuple[Decimal, ...]] = {}
    for selection in ("interior", "d_zero"):
        # Exactly one local refinement on each enumerated face.  The preceding
        # grid is deterministic seed selection, not falsely described as
        # multistart optimization.
        _, seed = best[selection][0]
        candidate = _bfgs(
            prepared, selection, _initial_coordinates(selection, *seed), truncation
        )
        candidates.append(candidate)
        refinement_objectives[selection] = (candidate.evaluation.objective,)

    eligible = [candidate for candidate in candidates if _eligible(candidate)]
    if not eligible:
        detail = ", ".join(
            f"{c.selection}: conv={c.converged} grad={max(abs(v) for v in c.transformed_gradient)}"
            for c in candidates
        )
        raise ArithmeticError(f"no attained KKT candidate ({detail})")
    minimum = min(candidate.evaluation.objective for candidate in eligible)
    tie = Decimal("1e-19") * max(ONE, abs(minimum))
    equivalent = [candidate for candidate in eligible if candidate.evaluation.objective <= minimum + tie]
    selected = min(equivalent, key=lambda candidate: 0 if candidate.selection == "d_zero" else 1)
    if selected.evaluation.objective > min(values[0][0] for values in best.values()) + tie:
        raise AssertionError("local refinement is worse than seed grid")
    seed_grid: dict[str, object] = {
        "evaluations": evaluations,
        "mu_offsets": mu_offsets,
        "alpha_values": alpha_values,
        "gamma_values": gamma_values,
        "beta_values": beta_values,
        "d_values": d_values,
        "face_seed_best_objectives": {key: value[0][0] for key, value in best.items()},
        "local_refinements_per_face": 1,
        "local_refinement_objectives": refinement_objectives,
        "optimizer": "Decimal inverse-BFGS with Armijo backtracking",
        "multistart_claimed": False,
    }
    return SolveResult(selected, tuple(candidates), seed_grid)


def _format_decimal(value: Decimal) -> str:
    if not value.is_finite():
        if value.is_nan():
            return "nan"
        return "+inf" if value > ZERO else "-inf"
    if abs(value) < NEAR_ZERO_ABSOLUTE_TOLERANCE:
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


def _candidate_record(candidate: Candidate, prepared: PreparedSeries) -> dict[str, object]:
    gradient = candidate.evaluation.physical_gradient
    return {
        "selection": candidate.selection,
        "mu_normalized": candidate.mu,
        "mu_physical": candidate.mu + TWO * prepared.scale.ln(),
        "omega_normalized": candidate.omega,
        "omega_physical": candidate.omega + TWO * (ONE - candidate.beta) * prepared.scale.ln(),
        "alpha": candidate.alpha,
        "gamma": candidate.gamma,
        "beta": candidate.beta,
        "d": candidate.d,
        "beta_lower_slack": ONE + candidate.beta,
        "beta_upper_slack": ONE - candidate.beta,
        "d_upper_slack": HALF - candidate.d,
        "objective": candidate.evaluation.objective,
        "physical_gradient": {
            "omega": gradient[0], "alpha": gradient[1], "gamma": gradient[2],
            "beta": gradient[3], "d": gradient[4],
        },
        "transformed_gradient": candidate.transformed_gradient,
        "transformed_gradient_norm": max(abs(value) for value in candidate.transformed_gradient),
        "iterations": candidate.iterations,
        "converged": candidate.converged,
        "open_boundary_guard_rejected": candidate.open_boundary_guard_rejected,
        "d_zero_directional_derivative": candidate.d_zero_directional_derivative,
    }


def _common(spec: CaseSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "purpose": spec.purpose,
        "observations": list(spec.observations),
        "truncation": spec.truncation,
        "expected_selection": spec.expected_selection,
        "expected_property": spec.expected_property,
        "reflection_of": spec.reflection_of,
    }


def _success_record(
    spec: CaseSpec, prepared: PreparedSeries, solved: SolveResult,
) -> dict[str, object]:
    selected = solved.selected
    return {
        **_common(spec),
        "outcome": "success",
        "mean": prepared.mean,
        "scale": prepared.scale,
        "initial_log_variance": prepared.initial_log_variance,
        "fit": _candidate_record(selected, prepared),
        "normalized_log_variances": selected.evaluation.log_variances,
        "physical_variances": selected.evaluation.physical_variances,
        "fractional_terms": selected.evaluation.fractional_terms,
        "one_step_forecast": {
            "normalized_log_variance": selected.evaluation.one_step_log_variance,
            "physical_log_variance": selected.evaluation.one_step_log_variance + TWO * prepared.scale.ln(),
            "physical_variance": selected.evaluation.one_step_physical_variance,
        },
        "face_candidates": [_candidate_record(candidate, prepared) for candidate in solved.face_candidates],
        "seed_grid": solved.seed_grid,
    }


def _failure_record(spec: CaseSpec, prepared: PreparedSeries) -> dict[str, object]:
    return {
        **_common(spec), "outcome": "unidentified_constant_series",
        "mean": prepared.mean, "scale": prepared.scale,
        "failure": "demeaned maximum absolute residual is zero",
    }


def _probe_record(spec: CaseSpec, prepared: PreparedSeries) -> dict[str, object]:
    if spec.probe_parameters is None:
        raise AssertionError("missing probe parameters")
    mu, alpha, gamma, beta, d = map(Decimal, spec.probe_parameters)
    evaluation = _evaluate(
        prepared, mu, alpha, gamma, beta, d, spec.truncation
    )
    return {
        **_common(spec),
        "outcome": (
            "extended_real_probe"
            if spec.expected_property == "extended_real"
            else "parameter_probe"
        ),
        "mean": prepared.mean, "scale": prepared.scale,
        "initial_log_variance": prepared.initial_log_variance,
        "parameters": {
            "mu_normalized": mu, "omega_normalized": (ONE - beta) * mu,
            "alpha": alpha, "gamma": gamma, "beta": beta, "d": d,
        },
        "decimal_objective": evaluation.objective,
        "decimal_objective_is_finite": evaluation.objective.is_finite(),
        "normalized_log_variances": evaluation.log_variances,
        "one_step_forecast": {
            "normalized_log_variance": evaluation.one_step_log_variance,
            "physical_variance": evaluation.one_step_physical_variance,
        },
    }


def _direct_and_identity_audit(
    spec: CaseSpec, prepared: PreparedSeries, fit: dict[str, object],
) -> tuple[Decimal, Decimal]:
    mu, alpha, gamma, beta, d = (
        fit["mu_normalized"], fit["alpha"], fit["gamma"], fit["beta"], fit["d"]
    )
    assert all(isinstance(value, Decimal) for value in (mu, alpha, gamma, beta, d))
    evaluation = _evaluate(
        prepared, mu, alpha, gamma, beta, d, spec.truncation  # type: ignore[arg-type]
    )
    objective, path, fractions, forecast = _objective_and_path_direct(
        prepared, mu, alpha, gamma, beta, d, spec.truncation  # type: ignore[arg-type]
    )
    errors = [abs(objective - evaluation.objective), abs(forecast - evaluation.one_step_log_variance)]
    errors.extend(abs(a - b) for a, b in zip(path, evaluation.log_variances))
    errors.extend(abs(a - b) for a, b in zip(fractions, evaluation.fractional_terms))
    direct_error = max(errors, default=ZERO)

    egarch_error = ZERO
    if d == ZERO:
        logh = prepared.initial_log_variance
        assert logh is not None
        c = Decimal(SQRT_2_OVER_PI_TEXT)
        egarch_path: list[Decimal] = []
        for x in prepared.normalized:
            egarch_path.append(logh)
            z = x * (-logh / TWO).exp()
            logh = (ONE - beta) * mu + beta * logh + alpha * (abs(z) - c) + gamma * z
        egarch_error = max(
            [abs(a - b) for a, b in zip(egarch_path, evaluation.log_variances)]
            + [abs(logh - evaluation.one_step_log_variance)],
            default=ZERO,
        )
    return direct_error, egarch_error


def _nonstationary_gradient_probes() -> dict[str, object]:
    """Fixed interior points whose derivatives are not optimizer KKT residuals."""
    near_spec = next(spec for spec in CASES if spec.name == "near_d_half")
    if near_spec.probe_parameters is None:
        raise AssertionError("near-d-half derivative probe is missing parameters")
    near_parameters = tuple(Decimal(value) for value in near_spec.probe_parameters)

    generic_spec = next(spec for spec in CASES if spec.name == "interior_long_memory")
    generic_prepared = _prepare(generic_spec.observations)
    if generic_prepared.initial_log_variance is None:
        raise AssertionError("generic derivative probe unexpectedly has zero scale")
    generic_parameters = (
        generic_prepared.initial_log_variance + Decimal("0.37"),
        Decimal("0.31"),
        Decimal("-0.27"),
        Decimal("-0.41"),
        Decimal("0.23"),
    )

    definitions = (
        ("near_d_half", near_spec, _prepare(near_spec.observations), near_parameters),
        ("generic_nonstationary", generic_spec, generic_prepared, generic_parameters),
    )
    records: dict[str, object] = {}
    gradient_keys = ("omega", "alpha", "gamma", "beta", "d")
    for name, spec, prepared, parameters in definitions:
        mu, alpha, gamma, beta, d = parameters
        evaluation = _evaluate(
            prepared, mu, alpha, gamma, beta, d, spec.truncation
        )
        gradient = dict(zip(gradient_keys, evaluation.physical_gradient))
        absolute_gradients = tuple(abs(value) for value in evaluation.physical_gradient)
        records[name] = {
            "series": spec.name,
            "truncation": spec.truncation,
            "parameters": {
                "mu": mu,
                "omega": (ONE - beta) * mu,
                "alpha": alpha,
                "gamma": gamma,
                "beta": beta,
                "d": d,
            },
            "physical_gradient": gradient,
            "minimum_absolute_gradient": min(absolute_gradients),
            "maximum_absolute_gradient": max(absolute_gradients),
            "difference_scheme": "centered in all five physical coordinates",
        }
    generic = records["generic_nonstationary"]
    assert isinstance(generic, dict)
    if generic["minimum_absolute_gradient"] <= Decimal("1e-3"):
        raise AssertionError("generic derivative audit point lacks material gradients")
    return records


def _assert_cross_case_properties(raw_cases: Sequence[dict[str, object]]) -> dict[str, object]:
    by_name = {str(case["name"]): case for case in raw_cases}
    base = by_name["interior_long_memory"]
    reflected = by_name["interior_long_memory_reflected"]
    d_zero = by_name["d_zero_boundary"]
    negative = by_name["negative_beta"]
    near = by_name["near_d_half"]
    fits = [case["fit"] for case in (base, reflected, d_zero, negative)]
    if any(not isinstance(fit, dict) for fit in fits):
        raise AssertionError("missing fit")
    base_fit, reflected_fit, zero_fit, negative_fit = fits
    assert all(isinstance(fit, dict) for fit in fits)
    if not (base_fit["d"] > Decimal("0.04")):
        raise AssertionError("long-memory fixture lost material d")
    if zero_fit["selection"] != "d_zero" or zero_fit["d"] != ZERO:
        raise AssertionError("d=0 face not selected")
    if not (negative_fit["beta"] < ZERO):
        raise AssertionError("negative-beta fixture did not recover beta < 0")
    near_parameters = near["parameters"]
    assert isinstance(near_parameters, dict)
    if not (Decimal("0.39") < near_parameters["d"] < Decimal("0.495")):
        raise AssertionError("near-half fixture did not recover high d")

    reflection_pairs = (
        (base_fit["mu_normalized"], reflected_fit["mu_normalized"]),
        (base_fit["omega_normalized"], reflected_fit["omega_normalized"]),
        (base_fit["alpha"], reflected_fit["alpha"]),
        (base_fit["beta"], reflected_fit["beta"]),
        (base_fit["d"], reflected_fit["d"]),
        (base_fit["objective"], reflected_fit["objective"]),
    )
    reflection_error = max(abs(a - b) for a, b in reflection_pairs)
    reflection_error = max(reflection_error, abs(base_fit["gamma"] + reflected_fit["gamma"]))
    reflection_error = max(
        reflection_error,
        max(abs(a - b) for a, b in zip(
            base["normalized_log_variances"], reflected["normalized_log_variances"]
        )),
    )
    if reflection_error > Decimal("1e-20"):
        raise AssertionError(f"reflection error {reflection_error}")

    # Positive scaling leaves normalized results unchanged and shifts physical
    # mu by exactly 2*ln(scale factor).
    factor = Decimal("7.25")
    scaled_observations = tuple(
        format(Decimal(value) * factor, "f") for value in CASES[0].observations
    )
    scaled = _prepare(scaled_observations)
    original = _prepare(CASES[0].observations)
    scale_normalized_error = max(
        abs(a - b) for a, b in zip(original.normalized, scaled.normalized)
    )
    physical_mu_shift_error = abs(
        (base_fit["mu_normalized"] + TWO * scaled.scale.ln())
        - (base_fit["mu_normalized"] + TWO * original.scale.ln() + TWO * factor.ln())
    )

    # Runtime K must matter away from d=0.
    prepared = original
    truncation_parameters = (
        base_fit["mu_normalized"], base_fit["alpha"], base_fit["gamma"],
        base_fit["beta"], base_fit["d"],
    )
    objective_k4 = _objective_only(prepared, *truncation_parameters, 4)
    objective_k18 = _objective_only(prepared, *truncation_parameters, 18)
    truncation_difference = abs(objective_k4 - objective_k18)
    if truncation_difference <= Decimal("1e-12"):
        raise AssertionError("truncation probe is not sensitive to K")

    barriers = {
        "beta_minus_one": _objective_only(prepared, ZERO, ZERO, ZERO, -ONE, ZERO, 8),
        "beta_plus_one": _objective_only(prepared, ZERO, ZERO, ZERO, ONE, ZERO, 8),
        "d_negative": _objective_only(prepared, ZERO, ZERO, ZERO, ZERO, Decimal("-0.01"), 8),
        "d_half": _objective_only(prepared, ZERO, ZERO, ZERO, ZERO, HALF, 8),
    }
    if any(value != Decimal("Infinity") for value in barriers.values()):
        raise AssertionError("domain barrier did not return ordered +infinity")

    direct_error = ZERO
    egarch_error = ZERO
    for case in (base, reflected, d_zero, negative):
        spec = next(item for item in CASES if item.name == case["name"])
        prepared_case = _prepare(spec.observations)
        fit = case["fit"]
        assert isinstance(fit, dict)
        direct, egarch = _direct_and_identity_audit(spec, prepared_case, fit)
        direct_error = max(direct_error, direct)
        egarch_error = max(egarch_error, egarch)

    return {
        "material_d_lower_bound": Decimal("0.04"),
        "negative_beta_recovered": True,
        "near_half_d_interval": (Decimal("0.39"), Decimal("0.495")),
        "reflection_max_absolute_error": reflection_error,
        "scale_normalized_max_absolute_error": scale_normalized_error,
        "physical_mu_scale_shift_absolute_error": physical_mu_shift_error,
        "d_zero_egarch_max_absolute_error": egarch_error,
        "independent_direct_recurrence_max_absolute_error": direct_error,
        "truncation_probe": {
            "short_k": 4, "long_k": 18,
            "short_objective": objective_k4, "long_objective": objective_k18,
            "absolute_difference": truncation_difference,
        },
        "positive_infinity_barriers": barriers,
        "nonstationary_gradient_probes": _nonstationary_gradient_probes(),
    }


def _build_raw(precision: int) -> dict[str, object]:
    with localcontext() as ctx:
        ctx.prec = precision
        ctx.Emax = 999_999_999
        ctx.Emin = -999_999_999
        constant = _sqrt_2_over_pi()
        raw_cases: list[dict[str, object]] = []
        for spec in CASES:
            prepared = _prepare(spec.observations)
            if spec.expected_selection == "unidentified":
                raw_cases.append(_failure_record(spec, prepared))
            elif spec.expected_selection == "probe":
                raw_cases.append(_probe_record(spec, prepared))
            else:
                solved = _solve(prepared, spec.truncation)
                if solved.selected.selection != spec.expected_selection:
                    raise AssertionError(
                        f"{spec.name}: selected {solved.selected.selection}, expected {spec.expected_selection}"
                    )
                raw_cases.append(_success_record(spec, prepared, solved))
        cross_checks = _assert_cross_case_properties(raw_cases)
        return {
            "schema_version": 1,
            "oracle": "independent Decimal fixed-K FIEGARCH(1,d,1) Gaussian QML",
            "provenance": {
                "reference": "Bollerslev and Mikkelsen (1996), Eq. 11",
                "reference_note": "finite fractional truncation and fixed sample initialization are explicit oracle choices",
                "generator": "scripts/fiegarch_qml_oracle.py",
                "emit_command": "python scripts/fiegarch_qml_oracle.py emit golden/fiegarch_qml.json",
                "dependencies": "Python standard library only",
            },
            "contract": {
                "centering": "demean observations",
                "normalization": "divide residuals by maximum absolute demeaned residual",
                "initialization": "L[0]=log(mean(normalized_residual^2)), fixed and parameter independent",
                "fractional_coefficients": "pi[0]=1; pi[k]=pi[k-1]*(k-1+d)/k, coefficients of (1-L)^(-d)",
                "news": "z[t]=x[t]*exp(-L[t]/2); g[t]=alpha*(abs(z[t])-sqrt(2/pi))+gamma*z[t]",
                "fractional_filter": "f[t]=sum(k=0..min(t,K-1), pi[k]*g[t-k]); K is runtime fixture data",
                "recurrence": "L[t+1]=(1-beta)*mu+beta*L[t]+f[t]",
                "objective": "0.5*sum(L[t]+x[t]^2*exp(-L[t])); Gaussian constant and n*log(scale) omitted",
                "domain": "alpha,gamma unrestricted; beta in (-1,1); d in [0,0.5); exact d=0 face plus interior",
                "physical_transform": "mu_physical=mu_normalized+2*log(scale); omega_physical=omega_normalized+2*(1-beta)*log(scale); alpha,gamma,beta,d unchanged",
            },
            "constants": {
                "sqrt_2_over_pi_embedded": SQRT_2_OVER_PI_TEXT,
                "sqrt_2_over_pi_decimal_audit": constant,
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
                "near_zero_absolute_floor": NEAR_ZERO_ABSOLUTE_FLOOR,
                "near_zero_absolute_tolerance": NEAR_ZERO_ABSOLUTE_TOLERANCE,
                "formatted_output_equality_required": True,
                "comparison": "true relative error away from the documented near-zero floor; absolute tolerance below it; raw Decimal trees use sorted-key traversal before formatting, then independently formatted trees must be exactly equal",
            },
            "binary64_replay": {
                "canonical_measurements": "none; host-dependent measurements are stdout-only",
                "extended_real_rule": "overflow and invalid-domain barriers are ordered +infinity, never NaN or an exception",
            },
        }


def _compare_raw(primary: object, verification: object, path: str = "") -> Decimal:
    if isinstance(primary, Decimal):
        if not isinstance(verification, Decimal):
            raise AssertionError(f"{path}: Decimal/type mismatch")
        if not primary.is_finite() or not verification.is_finite():
            if primary != verification:
                raise AssertionError(f"{path}: nonfinite mismatch")
            return ZERO
        error = abs(primary - verification)
        magnitude = max(abs(primary), abs(verification))
        if magnitude <= NEAR_ZERO_ABSOLUTE_FLOOR:
            tolerance = NEAR_ZERO_ABSOLUTE_TOLERANCE
            scale = NEAR_ZERO_ABSOLUTE_FLOOR
        else:
            scale = magnitude
            tolerance = (
                Decimal(10) ** Decimal(-AGREEMENT_SIGNIFICANT_DIGITS) * magnitude
            )
        if error > tolerance:
            raise AssertionError(f"{path}: raw 80/120 mismatch {error} > {tolerance}")
        return error / scale
    if isinstance(primary, dict):
        if not isinstance(verification, dict) or sorted(primary) != sorted(verification):
            raise AssertionError(f"{path}: mapping mismatch")
        return max((
            _compare_raw(primary[key], verification[key], f"{path}/{key}")
            for key in sorted(primary)
        ), default=ZERO)
    if isinstance(primary, (tuple, list)):
        if not isinstance(verification, type(primary)) or len(primary) != len(verification):
            raise AssertionError(f"{path}: sequence mismatch")
        return max((
            _compare_raw(a, b, f"{path}/{index}")
            for index, (a, b) in enumerate(zip(primary, verification))
        ), default=ZERO)
    if type(primary) is not type(verification) or primary != verification:
        raise AssertionError(f"{path}: scalar mismatch")
    return ZERO


def _gradient_audit(raw: dict[str, object]) -> Decimal:
    maximum = ZERO
    step = Decimal("1e-17")
    specs = {spec.name: spec for spec in CASES}
    keys = ("omega", "alpha", "gamma", "beta", "d")

    def audit_point(
        label: str,
        prepared: PreparedSeries,
        truncation: int,
        base: Sequence[Decimal],
        analytic: Sequence[Decimal],
        one_sided_d: bool,
    ) -> None:
        nonlocal maximum
        def physical_objective(values: Sequence[Decimal]) -> Decimal:
            omega, alpha, gamma, beta, d = values
            if not (-ONE < beta < ONE):
                return Decimal("Infinity")
            mu = omega / (ONE - beta)
            return _objective_only(
                prepared, mu, alpha, gamma, beta, d, truncation
            )

        for index, key in enumerate(keys):
            plus = list(base)
            minus = list(base)
            plus[index] += step
            minus[index] -= step
            if key == "d" and one_sided_d:
                plus_two = list(base)
                plus_two[index] += TWO * step
                numeric = (
                    -Decimal(3) * physical_objective(base)
                    + Decimal(4) * physical_objective(plus)
                    - physical_objective(plus_two)
                ) / (TWO * step)
            else:
                numeric = (physical_objective(plus) - physical_objective(minus)) / (TWO * step)
            error = abs(numeric - analytic[index])
            maximum = max(maximum, error)
            if error > GRADIENT_AUDIT_TOLERANCE:
                raise AssertionError(f"{label}/{key}: gradient error {error}")

    # Optimizer results remain useful regression points, including the exact
    # d=0 face where only the d coordinate requires a one-sided stencil.
    cases = raw["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        if case["outcome"] != "success":
            continue
        spec = specs[str(case["name"])]
        prepared = _prepare(spec.observations)
        fit = case["fit"]
        assert isinstance(fit, dict)
        base = tuple(
            fit[key]
            for key in ("omega_normalized", "alpha", "gamma", "beta", "d")
        )
        physical_gradient = fit["physical_gradient"]
        assert isinstance(physical_gradient, dict)
        analytic = tuple(physical_gradient[key] for key in keys)
        audit_point(
            spec.name, prepared, spec.truncation, base, analytic,
            one_sided_d=base[4] == ZERO,
        )

    # These two points make the audit non-vacuous: neither is an optimizer
    # stationary point, near_d_half exercises the open upper d boundary, and
    # generic_nonstationary has material derivatives in every coordinate.
    probes = _nonstationary_gradient_probes()
    for name in ("near_d_half", "generic_nonstationary"):
        probe = probes[name]
        assert isinstance(probe, dict)
        spec = specs[str(probe["series"])]
        prepared = _prepare(spec.observations)
        parameters = probe["parameters"]
        gradient = probe["physical_gradient"]
        assert isinstance(parameters, dict) and isinstance(gradient, dict)
        base = tuple(parameters[key] for key in ("omega", "alpha", "gamma", "beta", "d"))
        analytic = tuple(gradient[key] for key in keys)
        audit_point(
            name, prepared, spec.truncation, base, analytic,
            one_sided_d=False,
        )
    return maximum


def _float_replay_measurements(raw: dict[str, object]) -> dict[str, object]:
    maxima: dict[str, object] = {
        "objective_absolute": 0.0,
        "log_variance_absolute": 0.0,
        "extreme_probe_positive_infinity": False,
    }
    specs = {spec.name: spec for spec in CASES}
    cases = raw["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        if case["outcome"] not in ("success", "extended_real_probe"):
            continue
        spec = specs[str(case["name"])]
        observations = [float(value) for value in spec.observations]
        mean = math.fsum(observations) / len(observations)
        residuals = [value - mean for value in observations]
        scale = max(abs(value) for value in residuals)
        normalized = [value / scale for value in residuals]
        logh = math.log(math.fsum(value * value for value in normalized) / len(normalized))
        fit = case["fit"] if case["outcome"] == "success" else case["parameters"]
        assert isinstance(fit, dict)
        mu, alpha, gamma, beta, d = map(float, (
            fit["mu_normalized"], fit["alpha"], fit["gamma"], fit["beta"], fit["d"]
        ))
        coefficients = [1.0]
        for k in range(1, spec.truncation):
            coefficients.append(coefficients[-1] * (k - 1 + d) / k)
        news: list[float] = []
        path: list[float] = []
        objective = 0.0
        for x in normalized:
            try:
                standardized_square = x * x * math.exp(-logh)
            except OverflowError:
                standardized_square = math.inf
            objective += 0.5 * (logh + standardized_square)
            path.append(logh)
            try:
                z = x * math.exp(-0.5 * logh)
            except OverflowError:
                z = math.copysign(math.inf, x)
            news.append(alpha * (abs(z) - float(SQRT_2_OVER_PI_TEXT)) + gamma * z)
            upto = min(len(news), spec.truncation)
            filtered = math.fsum(coefficients[k] * news[-1-k] for k in range(upto))
            logh = (1.0 - beta) * mu + beta * logh + filtered
        if case["outcome"] == "extended_real_probe":
            if objective != math.inf:
                raise AssertionError("quiet/extreme binary64 replay did not reach +infinity")
            maxima["extreme_probe_positive_infinity"] = True
        else:
            maxima["objective_absolute"] = max(
                float(maxima["objective_absolute"]),
                abs(objective - float(fit["objective"])),
            )
            maxima["log_variance_absolute"] = max(
                float(maxima["log_variance_absolute"]),
                max(
                    abs(actual - float(expected))
                    for actual, expected in zip(path, case["normalized_log_variances"])
                ),
            )
    return maxima


TOP_KEYS = (
    "schema_version", "oracle", "provenance", "contract", "constants",
    "case_count", "cases", "cross_case_checks", "precision_recheck",
    "binary64_replay",
)
COMMON_CASE_KEYS = (
    "name", "purpose", "observations", "truncation", "expected_selection",
    "expected_property", "reflection_of", "outcome",
)
FIT_KEYS = (
    "selection", "mu_normalized", "mu_physical", "omega_normalized",
    "omega_physical", "alpha", "gamma", "beta", "d", "beta_lower_slack",
    "beta_upper_slack", "d_upper_slack", "objective", "physical_gradient",
    "transformed_gradient", "transformed_gradient_norm", "iterations",
    "converged", "open_boundary_guard_rejected",
    "d_zero_directional_derivative",
)


def _require_keys(value: object, keys: Sequence[str], path: str) -> dict[str, object]:
    if not isinstance(value, dict) or sorted(value) != sorted(keys):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise AssertionError(f"{path}: exact keys mismatch {actual!r}")
    return value


def _numeric_string(value: object, path: str, allow_nonfinite: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise AssertionError(f"{path}: numeric string required")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise AssertionError(f"{path}: invalid Decimal string") from exc
    if not allow_nonfinite and not parsed.is_finite():
        raise AssertionError(f"{path}: finite numeric string required")
    return parsed


def _numeric_list(value: object, path: str, length: int) -> list[object]:
    if not isinstance(value, list) or len(value) != length:
        raise AssertionError(f"{path}: list length {length} required")
    for index, item in enumerate(value):
        _numeric_string(item, f"{path}/{index}")
    return value


def _validate_fit(value: object, path: str) -> dict[str, object]:
    fit = _require_keys(value, FIT_KEYS, path)
    if fit["selection"] not in ("interior", "d_zero"):
        raise AssertionError(f"{path}/selection")
    numeric = (
        "mu_normalized", "mu_physical", "omega_normalized", "omega_physical",
        "alpha", "gamma", "beta", "d", "beta_lower_slack",
        "beta_upper_slack", "d_upper_slack", "objective",
        "transformed_gradient_norm",
    )
    for key in numeric:
        _numeric_string(fit[key], f"{path}/{key}")
    gradient = _require_keys(
        fit["physical_gradient"], ("omega", "alpha", "gamma", "beta", "d"),
        f"{path}/physical_gradient",
    )
    for key in gradient:
        _numeric_string(gradient[key], f"{path}/physical_gradient/{key}")
    expected_length = 5 if fit["selection"] == "interior" else 4
    _numeric_list(fit["transformed_gradient"], f"{path}/transformed_gradient", expected_length)
    if type(fit["iterations"]) is not int or fit["iterations"] < 0:
        raise AssertionError(f"{path}/iterations")
    for key in ("converged", "open_boundary_guard_rejected"):
        if type(fit[key]) is not bool:
            raise AssertionError(f"{path}/{key}")
    if fit["selection"] == "d_zero":
        _numeric_string(fit["d_zero_directional_derivative"], f"{path}/d_zero_directional_derivative")
    elif fit["d_zero_directional_derivative"] is not None:
        raise AssertionError(f"{path}/d_zero_directional_derivative")
    return fit


def _validate_seed_grid(value: object, path: str) -> None:
    keys = (
        "evaluations", "mu_offsets", "alpha_values", "gamma_values",
        "beta_values", "d_values", "face_seed_best_objectives",
        "local_refinements_per_face", "local_refinement_objectives",
        "optimizer", "multistart_claimed",
    )
    grid = _require_keys(value, keys, path)
    if grid["evaluations"] != 540 or type(grid["evaluations"]) is not int:
        raise AssertionError(f"{path}/evaluations")
    for key, length in (("mu_offsets", 3), ("alpha_values", 3), ("gamma_values", 3), ("beta_values", 4), ("d_values", 4)):
        _numeric_list(grid[key], f"{path}/{key}", length)
    for key in ("face_seed_best_objectives", "local_refinement_objectives"):
        block = _require_keys(grid[key], ("interior", "d_zero"), f"{path}/{key}")
        for face in block:
            if key == "local_refinement_objectives":
                _numeric_list(block[face], f"{path}/{key}/{face}", 1)
            else:
                _numeric_string(block[face], f"{path}/{key}/{face}")
    if grid["local_refinements_per_face"] != 1 or grid["multistart_claimed"] is not False:
        raise AssertionError(f"{path}: refinement semantics")
    if not isinstance(grid["optimizer"], str) or "BFGS" not in grid["optimizer"]:
        raise AssertionError(f"{path}/optimizer")


def _validate_schema(document: object) -> dict[str, object]:
    top = _require_keys(document, TOP_KEYS, "")
    if top["schema_version"] != 1 or type(top["schema_version"]) is not int:
        raise AssertionError("/schema_version")
    if not isinstance(top["oracle"], str):
        raise AssertionError("/oracle")
    provenance = _require_keys(
        top["provenance"],
        ("reference", "reference_note", "generator", "emit_command", "dependencies"),
        "/provenance",
    )
    if any(not isinstance(value, str) for value in provenance.values()):
        raise AssertionError("/provenance")
    if "Bollerslev and Mikkelsen (1996), Eq. 11" not in provenance["reference"]:
        raise AssertionError("/provenance/reference")
    contract_keys = (
        "centering", "normalization", "initialization", "fractional_coefficients",
        "news", "fractional_filter", "recurrence", "objective", "domain",
        "physical_transform",
    )
    contract = _require_keys(top["contract"], contract_keys, "/contract")
    if any(not isinstance(value, str) for value in contract.values()):
        raise AssertionError("/contract")
    constants = _require_keys(
        top["constants"],
        ("sqrt_2_over_pi_embedded", "sqrt_2_over_pi_decimal_audit",
         "kkt_tolerance", "open_boundary_guard", "output_significant_digits"),
        "/constants",
    )
    for key in tuple(constants)[:-1]:
        _numeric_string(constants[key], f"/constants/{key}")
    if constants["output_significant_digits"] != OUTPUT_SIGNIFICANT_DIGITS:
        raise AssertionError("/constants/output_significant_digits")
    if top["case_count"] != len(CASES) or type(top["case_count"]) is not int:
        raise AssertionError("/case_count")
    cases = top["cases"]
    if not isinstance(cases, list) or len(cases) != len(CASES):
        raise AssertionError("/cases")
    for index, (case, spec) in enumerate(zip(cases, CASES)):
        path = f"/cases/{index}"
        if not isinstance(case, dict):
            raise AssertionError(path)
        if [case.get("name"), case.get("truncation")] != [spec.name, spec.truncation]:
            raise AssertionError(f"{path}: identity")
        if case.get("observations") != list(spec.observations):
            raise AssertionError(f"{path}/observations")
        if any(not isinstance(value, str) for value in case["observations"]):
            raise AssertionError(f"{path}/observations type")
        outcome = case.get("outcome")
        if outcome == "success":
            expected_keys = COMMON_CASE_KEYS + (
                "mean", "scale", "initial_log_variance", "fit",
                "normalized_log_variances", "physical_variances", "fractional_terms",
                "one_step_forecast", "face_candidates", "seed_grid",
            )
            _require_keys(case, expected_keys, path)
            for key in ("mean", "scale", "initial_log_variance"):
                _numeric_string(case[key], f"{path}/{key}")
            fit = _validate_fit(case["fit"], f"{path}/fit")
            if fit["selection"] != spec.expected_selection:
                raise AssertionError(f"{path}/fit/selection")
            for key in ("normalized_log_variances", "physical_variances", "fractional_terms"):
                _numeric_list(case[key], f"{path}/{key}", len(spec.observations))
            forecast = _require_keys(
                case["one_step_forecast"],
                ("normalized_log_variance", "physical_log_variance", "physical_variance"),
                f"{path}/one_step_forecast",
            )
            for key in forecast:
                _numeric_string(forecast[key], f"{path}/one_step_forecast/{key}")
            if not isinstance(case["face_candidates"], list) or len(case["face_candidates"]) != 2:
                raise AssertionError(f"{path}/face_candidates")
            for face_index, candidate in enumerate(case["face_candidates"]):
                _validate_fit(candidate, f"{path}/face_candidates/{face_index}")
            if [item["selection"] for item in case["face_candidates"]] != ["interior", "d_zero"]:
                raise AssertionError(f"{path}/face_candidates order")
            _validate_seed_grid(case["seed_grid"], f"{path}/seed_grid")
        elif outcome == "unidentified_constant_series":
            _require_keys(case, COMMON_CASE_KEYS + ("mean", "scale", "failure"), path)
            _numeric_string(case["mean"], f"{path}/mean")
            _numeric_string(case["scale"], f"{path}/scale")
            if not isinstance(case["failure"], str):
                raise AssertionError(f"{path}/failure")
        elif outcome in ("extended_real_probe", "parameter_probe"):
            expected_keys = COMMON_CASE_KEYS + (
                "mean", "scale", "initial_log_variance", "parameters",
                "decimal_objective", "decimal_objective_is_finite",
                "normalized_log_variances", "one_step_forecast",
            )
            _require_keys(case, expected_keys, path)
            for key in ("mean", "scale", "initial_log_variance", "decimal_objective"):
                _numeric_string(case[key], f"{path}/{key}")
            if case["decimal_objective_is_finite"] is not True:
                raise AssertionError(f"{path}/decimal_objective_is_finite")
            parameters = _require_keys(
                case["parameters"],
                ("mu_normalized", "omega_normalized", "alpha", "gamma", "beta", "d"),
                f"{path}/parameters",
            )
            for key in parameters:
                _numeric_string(parameters[key], f"{path}/parameters/{key}")
            _numeric_list(
                case["normalized_log_variances"],
                f"{path}/normalized_log_variances", len(spec.observations),
            )
            forecast = _require_keys(
                case["one_step_forecast"],
                ("normalized_log_variance", "physical_variance"),
                f"{path}/one_step_forecast",
            )
            for key in forecast:
                _numeric_string(forecast[key], f"{path}/one_step_forecast/{key}", allow_nonfinite=True)
        else:
            raise AssertionError(f"{path}/outcome")

    cross = _require_keys(
        top["cross_case_checks"],
        ("material_d_lower_bound", "negative_beta_recovered", "near_half_d_interval",
         "reflection_max_absolute_error", "scale_normalized_max_absolute_error",
         "physical_mu_scale_shift_absolute_error", "d_zero_egarch_max_absolute_error",
         "independent_direct_recurrence_max_absolute_error", "truncation_probe",
         "positive_infinity_barriers", "nonstationary_gradient_probes"),
        "/cross_case_checks",
    )
    for key in (
        "material_d_lower_bound", "reflection_max_absolute_error",
        "scale_normalized_max_absolute_error", "physical_mu_scale_shift_absolute_error",
        "d_zero_egarch_max_absolute_error", "independent_direct_recurrence_max_absolute_error",
    ):
        _numeric_string(cross[key], f"/cross_case_checks/{key}")
    if cross["negative_beta_recovered"] is not True:
        raise AssertionError("/cross_case_checks/negative_beta_recovered")
    _numeric_list(cross["near_half_d_interval"], "/cross_case_checks/near_half_d_interval", 2)
    truncation = _require_keys(
        cross["truncation_probe"],
        ("short_k", "long_k", "short_objective", "long_objective", "absolute_difference"),
        "/cross_case_checks/truncation_probe",
    )
    if (truncation["short_k"], truncation["long_k"]) != (4, 18):
        raise AssertionError("/cross_case_checks/truncation_probe/K")
    for key in ("short_objective", "long_objective", "absolute_difference"):
        _numeric_string(truncation[key], f"/cross_case_checks/truncation_probe/{key}")
    barriers = _require_keys(
        cross["positive_infinity_barriers"],
        ("beta_minus_one", "beta_plus_one", "d_negative", "d_half"),
        "/cross_case_checks/positive_infinity_barriers",
    )
    for key, value in barriers.items():
        if _numeric_string(value, f"/cross_case_checks/positive_infinity_barriers/{key}", True) != Decimal("Infinity"):
            raise AssertionError(f"/cross_case_checks/positive_infinity_barriers/{key}")
    probes = _require_keys(
        cross["nonstationary_gradient_probes"],
        ("near_d_half", "generic_nonstationary"),
        "/cross_case_checks/nonstationary_gradient_probes",
    )
    probe_keys = (
        "series", "truncation", "parameters", "physical_gradient",
        "minimum_absolute_gradient", "maximum_absolute_gradient",
        "difference_scheme",
    )
    for name, probe_value in probes.items():
        probe_path = f"/cross_case_checks/nonstationary_gradient_probes/{name}"
        probe = _require_keys(probe_value, probe_keys, probe_path)
        if not isinstance(probe["series"], str):
            raise AssertionError(f"{probe_path}/series")
        if type(probe["truncation"]) is not int or probe["truncation"] < 1:
            raise AssertionError(f"{probe_path}/truncation")
        parameters = _require_keys(
            probe["parameters"], ("mu", "omega", "alpha", "gamma", "beta", "d"),
            f"{probe_path}/parameters",
        )
        gradient = _require_keys(
            probe["physical_gradient"], ("omega", "alpha", "gamma", "beta", "d"),
            f"{probe_path}/physical_gradient",
        )
        for key, value in parameters.items():
            _numeric_string(value, f"{probe_path}/parameters/{key}")
        for key, value in gradient.items():
            _numeric_string(value, f"{probe_path}/physical_gradient/{key}")
        minimum = _numeric_string(
            probe["minimum_absolute_gradient"],
            f"{probe_path}/minimum_absolute_gradient",
        )
        _numeric_string(
            probe["maximum_absolute_gradient"],
            f"{probe_path}/maximum_absolute_gradient",
        )
        if probe["difference_scheme"] != "centered in all five physical coordinates":
            raise AssertionError(f"{probe_path}/difference_scheme")
        if name == "generic_nonstationary" and minimum <= Decimal("1e-3"):
            raise AssertionError(f"{probe_path}: gradients are not material")
    precision = _require_keys(
        top["precision_recheck"],
        ("primary_decimal_digits", "verification_decimal_digits",
         "raw_agreement_significant_digits", "near_zero_absolute_floor",
         "near_zero_absolute_tolerance", "formatted_output_equality_required",
         "comparison"),
        "/precision_recheck",
    )
    if (
        precision["primary_decimal_digits"], precision["verification_decimal_digits"],
        precision["raw_agreement_significant_digits"],
    ) != (ORACLE_PRECISION, VERIFICATION_PRECISION, AGREEMENT_SIGNIFICANT_DIGITS):
        raise AssertionError("/precision_recheck")
    if _numeric_string(
        precision["near_zero_absolute_floor"],
        "/precision_recheck/near_zero_absolute_floor",
    ) != NEAR_ZERO_ABSOLUTE_FLOOR:
        raise AssertionError("/precision_recheck/near_zero_absolute_floor")
    if _numeric_string(
        precision["near_zero_absolute_tolerance"],
        "/precision_recheck/near_zero_absolute_tolerance",
    ) != NEAR_ZERO_ABSOLUTE_TOLERANCE:
        raise AssertionError("/precision_recheck/near_zero_absolute_tolerance")
    if precision["formatted_output_equality_required"] is not True:
        raise AssertionError("/precision_recheck/formatted_output_equality_required")
    if not isinstance(precision["comparison"], str):
        raise AssertionError("/precision_recheck/comparison")
    replay = _require_keys(
        top["binary64_replay"], ("canonical_measurements", "extended_real_rule"),
        "/binary64_replay",
    )
    if any(not isinstance(value, str) for value in replay.values()):
        raise AssertionError("/binary64_replay")
    return top


def _canonical_bytes(document: dict[str, object]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _deep_build() -> tuple[dict[str, object], Decimal, Decimal, dict[str, object]]:
    primary = _build_raw(ORACLE_PRECISION)
    verification = _build_raw(VERIFICATION_PRECISION)
    with localcontext() as ctx:
        ctx.prec = 140
        ctx.Emax = 999_999_999
        ctx.Emin = -999_999_999
        agreement = _compare_raw(primary, verification)
        gradient_error = _gradient_audit(verification)
        binary_errors = _float_replay_measurements(verification)
    formatted_primary = _format_tree(primary)
    formatted_verification = _format_tree(verification)
    if formatted_primary != formatted_verification:
        def first_difference(left: object, right: object, path: str = "") -> str:
            if isinstance(left, dict) and isinstance(right, dict):
                for key in sorted(left):
                    if left[key] != right[key]:
                        return first_difference(left[key], right[key], f"{path}/{key}")
            elif isinstance(left, list) and isinstance(right, list):
                for index, (a, b) in enumerate(zip(left, right)):
                    if a != b:
                        return first_difference(a, b, f"{path}/{index}")
            return f"{path}: {left!r} != {right!r}"
        raise AssertionError(
            "80/120 Decimal trees differ after independent canonical formatting at "
            + first_difference(formatted_primary, formatted_verification)
        )
    assert isinstance(formatted_primary, dict)
    _validate_schema(formatted_primary)
    return formatted_primary, agreement, gradient_error, binary_errors


def _load(path: Path) -> tuple[dict[str, object], bytes]:
    data = path.read_bytes()
    document = json.loads(data.decode("utf-8"))
    if not isinstance(document, dict):
        raise AssertionError("artifact root must be a mapping")
    return document, data


def emit(path: Path) -> None:
    started = time.perf_counter()
    document, agreement, gradient_error, binary_errors = _deep_build()
    data = _canonical_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(
        f"emitted {path} sha256={_sha256(data)} runtime={time.perf_counter()-started:.3f}s "
        f"raw_relative={agreement} gradient_abs={gradient_error} binary64={binary_errors}"
    )


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
    with localcontext() as ctx:
        ctx.prec = 60
        _sqrt_2_over_pi()
    print(f"fast-check ok: {path} sha256={digest}")


def deep_check(path: Path) -> None:
    started = time.perf_counter()
    expected, agreement, gradient_error, binary_errors = _deep_build()
    actual, data = _load(path)
    _validate_schema(actual)
    if actual != expected:
        raise AssertionError("artifact differs from regenerated 80/120 Decimal oracle")
    if data != _canonical_bytes(actual):
        raise AssertionError("artifact bytes are not canonical")
    digest = _sha256(data)
    if EXPECTED_ARTIFACT_SHA256 != "TO_BE_REPLACED_AFTER_DEEP_CHECK" and digest != EXPECTED_ARTIFACT_SHA256:
        raise AssertionError(f"artifact SHA-256 {digest} != pinned {EXPECTED_ARTIFACT_SHA256}")
    print(
        f"deep-check ok: sha256={digest} runtime={time.perf_counter()-started:.3f}s "
        f"raw_relative={agreement} gradient_abs={gradient_error} binary64={binary_errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "fast-check", "deep-check", "check"))
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("golden/fiegarch_qml.json")
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
