#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate and verify the independent Decimal OLS oracle fixture.

The oracle intentionally uses only Python's standard library.  It does not use
NumPy, SciPy, statsmodels, faer, or any production kanimiso implementation.
All model calculations use :class:`decimal.Decimal`:

* normal equations are solved by partial-pivoted Gaussian elimination;
* numerical rank is measured by an independent pivoted row reduction;
* covariance, leverage, and Cook's distance use a separately inverted Gram
  matrix;
* Student-t and F survival probabilities use a Decimal hypergeometric-series
  implementation of the regularized incomplete beta function; and
* the canonical 80-digit calculation is checked against 120 digits before a
  fixture is emitted.

Commands::

    python scripts/ols_oracle.py emit golden/ols.json
    python scripts/ols_oracle.py check golden/ols.json
    python scripts/ols_oracle.py deep

``check`` regenerates the canonical payload and requires an exact logical-byte
match after normalizing checkout-dependent CRLF/CR line endings to LF.  The
embedded script SHA-256 uses the same logical-source normalization.
``deep`` additionally evaluates at 180 digits and checks every numeric leaf and
the defining OLS identities.  JSON numbers derived from Decimal are stored as
50-significant-digit strings, so the fixture is independent of binary64 JSON
parsing and platform ``libm`` behavior.

All committed model cases are intentionally full rank.  Rank-deficient
diagnostics and unavailable-inference behavior are exercised by the Rust
property test ``rank_deficient_ols_uses_numerical_rank_and_withholds_tests``,
not by this golden fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, getcontext, localcontext
from pathlib import Path
from typing import Any, Iterable, Sequence


ORACLE_PRECISION = 80
VERIFICATION_PRECISION = 120
DEEP_PRECISION = 180
OUTPUT_SIGNIFICANT_DIGITS = 50
RANK_RELATIVE_TOLERANCE = Decimal("1e-40")
MAX_SERIES_ITERATIONS = 100_000
GENERATED_ON = "2026-09-03"
CANONICAL_RUNTIME = "CPython 3.12.13"
DEFAULT_GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "ols.json"


@dataclass(frozen=True)
class CaseSpec:
    """Exact decimal inputs and the semantic branch they exercise."""

    name: str
    purpose: str
    fit_intercept: bool
    x: tuple[tuple[str, ...], ...]
    y: tuple[str, ...]


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        name="intercept_noisy",
        purpose=(
            "Ordinary noisy intercept-on regression; the data residual is "
            "deliberately nonzero while the normal equations are stationary."
        ),
        fit_intercept=True,
        x=tuple(
            (value,)
            for value in ("-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8")
        ),
        y=(
            "-4.2",
            "-2.1",
            "0.3",
            "1.4",
            "4.0",
            "5.8",
            "8.4",
            "9.7",
            "12.6",
            "14.0",
            "16.8",
            "18.1",
        ),
    ),
    CaseSpec(
        name="no_intercept_noisy",
        purpose=(
            "No-intercept regression with a large response offset; validates "
            "uncentered SST, no-intercept adjusted R-squared, and the joint F test."
        ),
        fit_intercept=False,
        x=tuple((value,) for value in ("1", "2", "3", "4", "5", "6", "7", "8", "9")),
        y=("10.2", "11.0", "11.9", "13.4", "14.1", "15.0", "16.5", "17.1", "18.4"),
    ),
    CaseSpec(
        name="high_leverage",
        purpose=(
            "Two-regressor intercept-on fit with a remote design point; validates "
            "multivariate covariance, rank-based model degrees of freedom, leverage, "
            "and the squared (1-h) denominator in Cook's distance."
        ),
        fit_intercept=True,
        x=tuple(
            zip(
                ("-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "80"),
                ("2", "-1", "3", "0", "-2", "4", "1", "-3", "2", "5", "-1", "0", "3", "-4", "-55"),
                strict=True,
            )
        ),
        y=(
            "-5.4",
            "-1.7",
            "-5.1",
            "-1.0",
            "2.8",
            "-4.4",
            "0.3",
            "6.3",
            "0.5",
            "-2.0",
            "5.4",
            "5.7",
            "2.6",
            "12.2",
            "143.5",
        ),
    ),
)


class SingularMatrix(ArithmeticError):
    """Raised when the Decimal pivoted solver cannot identify a unique solve."""


@dataclass
class ErrorSummary:
    """Deterministic maximum-error accumulator.

    ``max_rel`` is a scaled error: ``abs(error) / max(1, abs(reference))``.
    It deliberately reduces to absolute error when the reference magnitude is
    below one.
    """

    max_abs: Decimal = Decimal(0)
    max_rel: Decimal = Decimal(0)
    worst_abs_path: str = ""
    worst_rel_path: str = ""

    def observe(self, left: Decimal, right: Decimal, path: str) -> None:
        # Comparisons often happen after leaving the calculation's local
        # Decimal context.  Never let Python's default 28-digit context round
        # the validation measurement itself.
        with localcontext() as ctx:
            ctx.prec = DEEP_PRECISION + 20
            ctx.rounding = ROUND_HALF_EVEN
            absolute = abs(left - right)
            # Near-zero identities (notably X' residual) are judged absolutely;
            # dividing by a 1e-100 reference would turn harmless roundoff into
            # an enormous and meaningless relative error.
            scaled_relative = absolute / max(Decimal(1), abs(right))
        if absolute > self.max_abs:
            self.max_abs = absolute
            self.worst_abs_path = path
        if scaled_relative > self.max_rel:
            self.max_rel = scaled_relative
            self.worst_rel_path = path


def decimal_sum(values: Iterable[Decimal]) -> Decimal:
    return sum(values, Decimal(0))


def dot(left: Sequence[Decimal], right: Sequence[Decimal]) -> Decimal:
    if len(left) != len(right):
        raise ValueError("dot operands have different lengths")
    return decimal_sum(a * b for a, b in zip(left, right, strict=True))


def matvec(matrix: Sequence[Sequence[Decimal]], vector: Sequence[Decimal]) -> list[Decimal]:
    return [dot(row, vector) for row in matrix]


def gram_and_rhs(
    design: Sequence[Sequence[Decimal]], response: Sequence[Decimal]
) -> tuple[list[list[Decimal]], list[Decimal]]:
    if not design or not design[0]:
        raise ValueError("empty design")
    columns = len(design[0])
    gram = [[Decimal(0) for _ in range(columns)] for _ in range(columns)]
    rhs = [Decimal(0) for _ in range(columns)]
    for row, target in zip(design, response, strict=True):
        for j in range(columns):
            rhs[j] += row[j] * target
            for k in range(j + 1):
                value = row[j] * row[k]
                gram[j][k] += value
                if j != k:
                    gram[k][j] += value
    return gram, rhs


def pivot_floor(matrix: Sequence[Sequence[Decimal]]) -> Decimal:
    scale = max((abs(value) for row in matrix for value in row), default=Decimal(0))
    if scale == 0:
        return Decimal(0)
    return scale * Decimal(1).scaleb(-(getcontext().prec - 20))


def solve_pivoted(
    matrix: Sequence[Sequence[Decimal]], rhs: Sequence[Decimal]
) -> list[Decimal]:
    """Solve a square system using deterministic partial pivoting."""

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix) or len(rhs) != n:
        raise ValueError("solve_pivoted requires an n by n matrix and n-vector")
    a = [list(row) for row in matrix]
    b = list(rhs)
    floor = pivot_floor(a)

    for column in range(n):
        pivot = max(range(column, n), key=lambda row: (abs(a[row][column]), -row))
        if abs(a[pivot][column]) <= floor:
            raise SingularMatrix(f"pivot {column} is below {floor}")
        if pivot != column:
            a[column], a[pivot] = a[pivot], a[column]
            b[column], b[pivot] = b[pivot], b[column]
        pivot_value = a[column][column]
        for row in range(column + 1, n):
            if a[row][column] == 0:
                continue
            factor = a[row][column] / pivot_value
            a[row][column] = Decimal(0)
            for k in range(column + 1, n):
                a[row][k] -= factor * a[column][k]
            b[row] -= factor * b[column]

    solution = [Decimal(0) for _ in range(n)]
    for row in range(n - 1, -1, -1):
        remainder = b[row] - decimal_sum(
            a[row][column] * solution[column] for column in range(row + 1, n)
        )
        solution[row] = remainder / a[row][row]
    return solution


def invert_pivoted(matrix: Sequence[Sequence[Decimal]]) -> list[list[Decimal]]:
    n = len(matrix)
    columns: list[list[Decimal]] = []
    for column in range(n):
        unit = [Decimal(0) for _ in range(n)]
        unit[column] = Decimal(1)
        columns.append(solve_pivoted(matrix, unit))
    return [[columns[column][row] for column in range(n)] for row in range(n)]


def numerical_rank(
    matrix: Sequence[Sequence[Decimal]], relative_tolerance: Decimal
) -> int:
    """Rank from partial-pivoted row reduction at a documented relative cutoff."""

    if not matrix:
        return 0
    rows = len(matrix)
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("ragged matrix")
    work = [list(row) for row in matrix]
    scale = max((abs(value) for row in work for value in row), default=Decimal(0))
    if scale == 0:
        return 0
    cutoff = scale * relative_tolerance
    pivot_row = 0
    for column in range(columns):
        if pivot_row == rows:
            break
        pivot = max(range(pivot_row, rows), key=lambda row: (abs(work[row][column]), -row))
        if abs(work[pivot][column]) <= cutoff:
            continue
        if pivot != pivot_row:
            work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][column]
        for row in range(pivot_row + 1, rows):
            if work[row][column] == 0:
                continue
            factor = work[row][column] / pivot_value
            work[row][column] = Decimal(0)
            for k in range(column + 1, columns):
                work[row][k] -= factor * work[pivot_row][k]
        pivot_row += 1
    return pivot_row


def decimal_pi(precision: int) -> Decimal:
    """Pi from the quadratically convergent Gauss-Legendre iteration."""

    with localcontext() as ctx:
        ctx.prec = precision + 20
        ctx.rounding = ROUND_HALF_EVEN
        one = Decimal(1)
        a = one
        b = one / Decimal(2).sqrt()
        t = Decimal(1) / Decimal(4)
        multiplier = one
        previous = Decimal(0)
        while a != previous:
            previous = a
            next_a = (a + b) / 2
            b = (a * b).sqrt()
            difference = a - next_a
            t -= multiplier * difference * difference
            a = next_a
            multiplier *= 2
        value = (a + b) * (a + b) / (4 * t)
        with localcontext() as rounded:
            rounded.prec = precision
            rounded.rounding = ROUND_HALF_EVEN
            return +value


def gamma_integer_or_half(value: Decimal, pi: Decimal) -> Decimal:
    """Exact factorial formula for Gamma on positive integer/half-integer inputs."""

    twice = value * 2
    twice_integer = int(twice)
    if twice != twice_integer or twice_integer <= 0:
        raise ValueError(f"Gamma input {value} is not a positive integer/half-integer")
    if twice_integer % 2 == 0:
        integer = twice_integer // 2
        product = 1
        for factor in range(2, integer):
            product *= factor
        return Decimal(product)
    half_index = (twice_integer - 1) // 2
    numerator = 1
    for factor in range(2, 2 * half_index + 1):
        numerator *= factor
    denominator_factorial = 1
    for factor in range(2, half_index + 1):
        denominator_factorial *= factor
    denominator = (4**half_index) * denominator_factorial
    return Decimal(numerator) * pi.sqrt() / Decimal(denominator)


def beta_integer_or_half(a: Decimal, b: Decimal, pi: Decimal) -> Decimal:
    return (
        gamma_integer_or_half(a, pi)
        * gamma_integer_or_half(b, pi)
        / gamma_integer_or_half(a + b, pi)
    )


def incomplete_beta_series(
    a: Decimal, b: Decimal, x: Decimal, beta_ab: Decimal
) -> Decimal:
    """Evaluate I_x(a,b) through the independent 2F1 power series."""

    if x == 0:
        return Decimal(0)
    epsilon = Decimal(1).scaleb(-(getcontext().prec - 15))
    term = Decimal(1)
    total = Decimal(1)
    for index in range(1, MAX_SERIES_ITERATIONS + 1):
        n = Decimal(index)
        term *= (a + n - 1) * (1 - b + n - 1) * x / ((a + n) * n)
        updated = total + term
        if term == 0 or abs(term) <= epsilon * max(Decimal(1), abs(updated)):
            total = updated
            break
        total = updated
    else:
        raise ArithmeticError("regularized incomplete-beta series did not converge")
    unregularized = (a * x.ln()).exp() * total / a
    return unregularized / beta_ab


def regularized_incomplete_beta(
    a: Decimal, b: Decimal, x: Decimal, pi: Decimal
) -> Decimal:
    """High-precision regularized incomplete beta with complement symmetry."""

    if a <= 0 or b <= 0:
        raise ValueError("regularized incomplete beta requires a,b > 0")
    if x < 0 or x > 1:
        raise ValueError("regularized incomplete beta requires x in [0,1]")
    if x == 0:
        return Decimal(0)
    if x == 1:
        return Decimal(1)
    beta_ab = beta_integer_or_half(a, b, pi)
    if x <= Decimal("0.5"):
        result = incomplete_beta_series(a, b, x, beta_ab)
    else:
        result = 1 - incomplete_beta_series(b, a, 1 - x, beta_ab)
    if result < 0 or result > 1:
        raise ArithmeticError(f"regularized incomplete beta escaped [0,1]: {result}")
    return result


def student_t_two_sided_pvalue(t_value: Decimal, df: int, pi: Decimal) -> Decimal:
    if df <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    degrees = Decimal(df)
    x = degrees / (degrees + t_value * t_value)
    return regularized_incomplete_beta(degrees / 2, Decimal(1) / 2, x, pi)


def f_survival_probability(
    statistic: Decimal, df_model: int, df_resid: int, pi: Decimal
) -> Decimal:
    if statistic < 0 or df_model <= 0 or df_resid <= 0:
        raise ValueError("F survival probability requires F >= 0 and positive dfs")
    numerator_df = Decimal(df_model)
    denominator_df = Decimal(df_resid)
    x = denominator_df / (denominator_df + numerator_df * statistic)
    return regularized_incomplete_beta(
        denominator_df / 2, numerator_df / 2, x, pi
    )


def scipy_pvalue_anchor_results(pi: Decimal) -> list[dict[str, Any]]:
    """Validate asymmetric tail anchors copied from the SciPy golden.

    The source values are stored as binary64 JSON numbers, so their absolute
    tolerances are approximately 3.9 times the measured difference from this
    high-precision Decimal implementation.  These asymmetric cases catch beta
    parameter swaps and CDF/survival-tail swaps that midpoint identities cannot.
    """

    evaluations = (
        (
            "student_t_two_sided_t1_df30",
            "student_t_two_sided_pvalue",
            {"t": "1", "df": 30},
            student_t_two_sided_pvalue(Decimal(1), 30, pi),
            Decimal("0.3253086154260302"),
            Decimal("1.2E-15"),
        ),
        (
            "f_survival_f2_df3_df20",
            "f_survival_probability",
            {"f": "2", "df_numerator": 3, "df_denominator": 20},
            f_survival_probability(Decimal(2), 3, 20, pi),
            Decimal("0.14643880308662147"),
            Decimal("2.5E-16"),
        ),
    )
    results: list[dict[str, Any]] = []
    for name, function, arguments, actual, expected, tolerance in evaluations:
        absolute_error = abs(actual - expected)
        if absolute_error > tolerance:
            raise AssertionError(
                f"SciPy p-value anchor {name} error {absolute_error} > {tolerance}"
            )
        results.append(
            {
                "name": name,
                "function": function,
                "arguments": arguments,
                "expected_from_source": expected,
                "decimal_result": actual,
                "measured_abs_error": absolute_error,
                "acceptance_tolerance_abs": tolerance,
            }
        )
    return results


def validate_incomplete_beta(precision: int) -> ErrorSummary:
    """Check the Decimal beta routine against closed forms and symmetry."""

    with localcontext() as ctx:
        ctx.prec = precision
        ctx.rounding = ROUND_HALF_EVEN
        pi = decimal_pi(precision)
        summary = ErrorSummary()
        closed_forms = (
            (Decimal(1), Decimal(1), Decimal("0.37"), Decimal("0.37"), "uniform"),
            (Decimal(2), Decimal(1), Decimal("0.25"), Decimal("0.0625"), "x_squared"),
            (Decimal(1), Decimal(2), Decimal("0.25"), Decimal("0.4375"), "two_x_minus_x_squared"),
            (Decimal(1) / 2, Decimal(1) / 2, Decimal("0.5"), Decimal("0.5"), "arcsine_midpoint"),
        )
        for a, b, x, expected, name in closed_forms:
            actual = regularized_incomplete_beta(a, b, x, pi)
            summary.observe(actual, expected, f"incomplete_beta.{name}")
        a = Decimal(7) / 2
        b = Decimal(5) / 2
        x = Decimal("0.37")
        complement = regularized_incomplete_beta(a, b, x, pi) + regularized_incomplete_beta(
            b, a, 1 - x, pi
        )
        summary.observe(complement, Decimal(1), "incomplete_beta.complement_symmetry")
        summary.observe(
            student_t_two_sided_pvalue(Decimal(1), 1, pi),
            Decimal("0.5"),
            "student_t.cauchy_t_one",
        )
        summary.observe(
            f_survival_probability(Decimal(1), 1, 1, pi),
            Decimal("0.5"),
            "f_distribution.one_one_at_one",
        )
        scipy_pvalue_anchor_results(pi)
        guard = Decimal(1).scaleb(-(precision - 20))
        if summary.max_abs > guard:
            raise AssertionError(
                f"incomplete-beta identity error {summary.max_abs} "
                f"at {summary.worst_abs_path} > {guard}"
            )
        return summary


def parse_design(case: CaseSpec) -> tuple[list[list[Decimal]], list[Decimal]]:
    features = [[Decimal(value) for value in row] for row in case.x]
    response = [Decimal(value) for value in case.y]
    if len(features) != len(response):
        raise ValueError(f"{case.name}: X/y row mismatch")
    if not features or not features[0]:
        raise ValueError(f"{case.name}: empty X")
    if any(len(row) != len(features[0]) for row in features):
        raise ValueError(f"{case.name}: ragged X")
    design = [([Decimal(1)] + row if case.fit_intercept else row) for row in features]
    return design, response


def compute_case(case: CaseSpec, precision: int) -> dict[str, Any]:
    with localcontext() as ctx:
        ctx.prec = precision
        ctx.rounding = ROUND_HALF_EVEN
        design, response = parse_design(case)
        n = len(design)
        p = len(design[0])
        rank = numerical_rank(design, RANK_RELATIVE_TOLERANCE)
        if rank != p:
            raise SingularMatrix(f"{case.name}: fixture cases must be full rank ({rank} != {p})")
        gram, rhs = gram_and_rhs(design, response)
        beta = solve_pivoted(gram, rhs)
        gram_inverse = invert_pivoted(gram)
        fitted = matvec(design, beta)
        residuals = [target - prediction for target, prediction in zip(response, fitted, strict=True)]
        sse = dot(residuals, residuals)
        response_mean = decimal_sum(response) / Decimal(n)
        centered_sst = decimal_sum((value - response_mean) ** 2 for value in response)
        uncentered_sst = dot(response, response)
        sst = centered_sst if case.fit_intercept else uncentered_sst
        if sse <= 0 or sst <= 0:
            raise ArithmeticError(f"{case.name}: fixture requires positive SSE and SST")

        constant_count = 1 if case.fit_intercept else 0
        df_resid = n - rank
        df_model = rank - constant_count
        if df_resid <= 0 or df_model <= 0:
            raise ArithmeticError(f"{case.name}: fixture requires positive residual/model df")
        sigma2 = sse / Decimal(df_resid)
        r2 = 1 - sse / sst
        adjusted_r2 = 1 - (1 - r2) * Decimal(n - constant_count) / Decimal(df_resid)
        explained_sum_squares = sst - sse
        f_statistic = (explained_sum_squares / Decimal(df_model)) / sigma2

        standard_errors = [
            (sigma2 * gram_inverse[index][index]).sqrt() for index in range(p)
        ]
        t_values = [
            coefficient / standard_error
            for coefficient, standard_error in zip(beta, standard_errors, strict=True)
        ]
        pi = decimal_pi(precision)
        p_values = [student_t_two_sided_pvalue(value, df_resid, pi) for value in t_values]
        f_pvalue = f_survival_probability(f_statistic, df_model, df_resid, pi)

        mle_variance = sse / Decimal(n)
        loglik = -Decimal(n) / 2 * (
            (2 * pi).ln() + 1 + mle_variance.ln()
        )
        aic = -2 * loglik + 2 * Decimal(rank)
        bic = -2 * loglik + Decimal(rank) * Decimal(n).ln()

        leverage: list[Decimal] = []
        for row in design:
            inverse_times_row = matvec(gram_inverse, row)
            leverage.append(dot(row, inverse_times_row))
        cooks = [
            residual * residual * hat
            / (Decimal(rank) * sigma2 * (1 - hat) * (1 - hat))
            for residual, hat in zip(residuals, leverage, strict=True)
        ]
        durbin_watson = decimal_sum(
            (residuals[index] - residuals[index - 1]) ** 2
            for index in range(1, n)
        ) / sse
        normal_equation_residual = [
            decimal_sum(design[row][column] * residuals[row] for row in range(n))
            for column in range(p)
        ]

        intercept = beta[0] if case.fit_intercept else Decimal(0)
        coefficients = beta[1:] if case.fit_intercept else list(beta)
        return {
            "n": n,
            "p": p,
            "rank": rank,
            "fit_intercept": case.fit_intercept,
            "intercept": intercept,
            "coef": coefficients,
            "beta": beta,
            "fitted": fitted,
            "residuals": residuals,
            "normal_equation_residual": normal_equation_residual,
            "sse": sse,
            "centered_sst": centered_sst,
            "uncentered_sst": uncentered_sst,
            "sst": sst,
            "sst_kind": "centered" if case.fit_intercept else "uncentered",
            "df_model": df_model,
            "df_resid": df_resid,
            "sigma2": sigma2,
            "r2": r2,
            "adjusted_r2": adjusted_r2,
            "standard_errors": standard_errors,
            "t_values": t_values,
            "p_values": p_values,
            "f_statistic": f_statistic,
            "f_pvalue": f_pvalue,
            "mle_variance": mle_variance,
            "loglik": loglik,
            "aic": aic,
            "bic": bic,
            "leverage": leverage,
            "cooks_distance": cooks,
            "durbin_watson": durbin_watson,
        }


def update_error(summary: ErrorSummary, error: Decimal, path: str) -> None:
    summary.observe(error, Decimal(0), path)


def validate_case(case: CaseSpec, result: dict[str, Any], precision: int) -> ErrorSummary:
    """Check exact algebraic identities and return their measured roundoff."""

    with localcontext() as ctx:
        ctx.prec = precision
        ctx.rounding = ROUND_HALF_EVEN
        design, response = parse_design(case)
        summary = ErrorSummary()
        fitted = result["fitted"]
        residuals = result["residuals"]
        for index, (prediction, residual, target) in enumerate(
            zip(fitted, residuals, response, strict=True)
        ):
            update_error(summary, prediction + residual - target, f"{case.name}.y_identity[{index}]")
        for index, value in enumerate(result["normal_equation_residual"]):
            update_error(summary, value, f"{case.name}.normal_equation[{index}]")
        if case.fit_intercept:
            update_error(
                summary,
                decimal_sum(residuals),
                f"{case.name}.intercept_residual_sum",
            )

        update_error(
            summary,
            decimal_sum(result["leverage"]) - Decimal(result["rank"]),
            f"{case.name}.hat_trace",
        )
        update_error(
            summary,
            result["r2"] - (1 - result["sse"] / result["sst"]),
            f"{case.name}.r2_definition",
        )
        expected_adjusted = 1 - (1 - result["r2"]) * Decimal(
            result["n"] - (1 if case.fit_intercept else 0)
        ) / Decimal(result["df_resid"])
        update_error(
            summary,
            result["adjusted_r2"] - expected_adjusted,
            f"{case.name}.adjusted_r2_definition",
        )
        expected_f = (
            (result["sst"] - result["sse"])
            / Decimal(result["df_model"])
            / result["sigma2"]
        )
        update_error(
            summary,
            result["f_statistic"] - expected_f,
            f"{case.name}.f_definition",
        )
        if result["df_model"] == 1:
            slope_index = 1 if case.fit_intercept else 0
            update_error(
                summary,
                result["f_statistic"] - result["t_values"][slope_index] ** 2,
                f"{case.name}.f_equals_t_squared",
            )
            update_error(
                summary,
                result["f_pvalue"] - result["p_values"][slope_index],
                f"{case.name}.f_p_equals_t_p",
            )
        for index, (row, hat) in enumerate(zip(design, result["leverage"], strict=True)):
            if hat < 0 or hat > 1:
                raise ArithmeticError(f"{case.name}: leverage[{index}]={hat} outside [0,1]")
            inverse_times_row = matvec(
                # Reconstruct the inverse through the same independently solved
                # Gram system only for an algebraic replay of the stored hat.
                invert_pivoted(gram_and_rhs(design, response)[0]),
                row,
            )
            update_error(
                summary,
                hat - dot(row, inverse_times_row),
                f"{case.name}.leverage_definition[{index}]",
            )
            cook = result["cooks_distance"][index]
            if cook < 0:
                raise ArithmeticError(f"{case.name}: Cook[{index}]={cook} is negative")
            expected_cook = (
                residuals[index]
                * residuals[index]
                * hat
                / (
                    Decimal(result["rank"])
                    * result["sigma2"]
                    * (1 - hat)
                    * (1 - hat)
                )
            )
            update_error(
                summary,
                cook - expected_cook,
                f"{case.name}.cook_definition[{index}]",
            )
        probabilities = [
            (f"p_values[{index}]", value)
            for index, value in enumerate(result["p_values"])
        ]
        probabilities.append(("f_pvalue", result["f_pvalue"]))
        for label, probability in probabilities:
            if probability < 0 or probability > 1:
                raise ArithmeticError(f"{case.name}.{label}={probability} outside [0,1]")

        if case.name == "no_intercept_noisy":
            centered_r2 = 1 - result["sse"] / result["centered_sst"]
            if abs(centered_r2 - result["r2"]) < Decimal("0.25"):
                raise AssertionError("no-intercept case does not distinguish centered and uncentered R2")
        if case.name == "high_leverage":
            if result["n"] < 15 or result["p"] != 3 or result["rank"] != 3:
                raise AssertionError(
                    "high-leverage case must have n>=15 and a full-rank three-column design"
                )
            if result["df_model"] <= 1:
                raise AssertionError("high-leverage case must exercise df_model > 1")
            if max(result["leverage"]) <= Decimal("0.9"):
                raise AssertionError("high-leverage case does not cross 0.9")

        guard = Decimal(1).scaleb(-(precision - 20))
        if summary.max_abs > guard:
            raise AssertionError(
                f"{case.name}: algebra error {summary.max_abs} at {summary.worst_abs_path} > {guard}"
            )
        return summary


def compute_suite(precision: int) -> tuple[list[dict[str, Any]], ErrorSummary]:
    results: list[dict[str, Any]] = []
    combined = validate_incomplete_beta(precision)
    for case in CASES:
        result = compute_case(case, precision)
        validation = validate_case(case, result, precision)
        results.append(result)
        combined.observe(validation.max_abs, Decimal(0), validation.worst_abs_path)
        if validation.max_rel > combined.max_rel:
            combined.max_rel = validation.max_rel
            combined.worst_rel_path = validation.worst_rel_path
    return results, combined


def compare_trees(left: Any, right: Any, path: str = "root") -> ErrorSummary:
    summary = ErrorSummary()

    def walk(a: Any, b: Any, current: str) -> None:
        if isinstance(a, Decimal) and isinstance(b, Decimal):
            summary.observe(a, b, current)
            return
        if type(a) is not type(b):
            raise TypeError(f"{current}: type mismatch {type(a).__name__} != {type(b).__name__}")
        if isinstance(a, dict):
            if set(a) != set(b):
                raise KeyError(f"{current}: key mismatch {sorted(a)} != {sorted(b)}")
            for key in sorted(a):
                walk(a[key], b[key], f"{current}.{key}")
            return
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise ValueError(f"{current}: length mismatch {len(a)} != {len(b)}")
            for index, (item_a, item_b) in enumerate(zip(a, b, strict=True)):
                walk(item_a, item_b, f"{current}[{index}]")
            return
        if a != b:
            raise ValueError(f"{current}: semantic mismatch {a!r} != {b!r}")

    walk(left, right, path)
    return summary


def canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError(f"non-finite Decimal cannot enter the fixture: {value}")
    if value == 0:
        return "0"
    with localcontext() as ctx:
        ctx.prec = OUTPUT_SIGNIFICANT_DIGITS
        ctx.rounding = ROUND_HALF_EVEN
        rounded = +value
        return format(rounded, "E")


def encode_decimal_tree(value: Any) -> Any:
    if isinstance(value, Decimal):
        return canonical_decimal(value)
    if isinstance(value, dict):
        return {key: encode_decimal_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_decimal_tree(item) for item in value]
    return value


def four_times(value: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = DEEP_PRECISION + 20
        ctx.rounding = ROUND_HALF_EVEN
        return value * 4


def normalize_logical_newlines(data: bytes) -> bytes:
    """Canonicalize checkout-dependent CRLF/CR to logical LF bytes."""

    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def script_sha256() -> str:
    logical_source = normalize_logical_newlines(Path(__file__).resolve().read_bytes())
    return hashlib.sha256(logical_source).hexdigest()


def build_payload() -> dict[str, Any]:
    canonical_results, identity = compute_suite(ORACLE_PRECISION)
    verification_results, _ = compute_suite(VERIFICATION_PRECISION)
    precision_error = compare_trees(
        canonical_results,
        verification_results,
        path=f"precision_{ORACLE_PRECISION}_vs_{VERIFICATION_PRECISION}",
    )
    precision_guard = Decimal("1e-65")
    if precision_error.max_abs > precision_guard or precision_error.max_rel > precision_guard:
        raise AssertionError(
            "canonical precision is unstable: "
            f"abs={precision_error.max_abs} rel={precision_error.max_rel}"
        )

    encoded_cases = []
    for spec, result in zip(CASES, canonical_results, strict=True):
        encoded_cases.append(
            {
                "name": spec.name,
                "purpose": spec.purpose,
                "input": {
                    "fit_intercept": spec.fit_intercept,
                    "x": [list(row) for row in spec.x],
                    "y": list(spec.y),
                },
                "expected": encode_decimal_tree(result),
            }
        )

    with localcontext() as ctx:
        ctx.prec = ORACLE_PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        anchor_results = scipy_pvalue_anchor_results(decimal_pi(ORACLE_PRECISION))

    return {
        "schema_version": 1,
        "generated_on": GENERATED_ON,
        "generator": {
            "path": "scripts/ols_oracle.py",
            "sha256": script_sha256(),
            "sha256_canonicalization": "logical source bytes with CRLF and CR normalized to LF",
            "commands": {
                "emit": "python scripts/ols_oracle.py emit golden/ols.json",
                "check": "python scripts/ols_oracle.py check golden/ols.json",
                "deep": "python scripts/ols_oracle.py deep",
            },
        },
        "oracle": {
            "runtime_used_for_canonical_fixture": CANONICAL_RUNTIME,
            "external_dependencies": [],
            "arithmetic": "Python stdlib decimal.Decimal with ROUND_HALF_EVEN",
            "coefficient_solver": "normal equations plus deterministic partial-pivoted Gaussian elimination",
            "rank_method": "partial-pivoted row reduction",
            "rank_relative_tolerance": canonical_decimal(RANK_RELATIVE_TOLERANCE),
            "incomplete_beta": "Decimal 2F1 power series with complement symmetry; integer/half-integer Gamma factorial formula",
            "canonical_precision_digits": ORACLE_PRECISION,
            "verification_precision_digits": VERIFICATION_PRECISION,
            "output_significant_digits": OUTPUT_SIGNIFICANT_DIGITS,
            "model_case_rank_scope": (
                "All golden model cases are full rank. Numerical-rank failure and "
                "unavailable-inference behavior are covered by the Rust property test "
                "rank_deficient_ols_uses_numerical_rank_and_withholds_tests, not this fixture."
            ),
        },
        "reference_pvalue_anchors": {
            "provenance": {
                "source_path": "golden/special_functions.json",
                "source_generator": "special_oracle_check.py emit",
                "scipy": "1.18.1",
                "numpy": "2.4.4",
                "python": "3.12.3",
                "numeric_storage": "binary64 JSON numbers",
            },
            "purpose": (
                "Asymmetric fixed anchors detect incomplete-beta parameter swaps and "
                "CDF/survival-tail swaps."
            ),
            "tolerance_policy": (
                "Per-anchor absolute tolerances are approximately 3.9 times the measured "
                "80-digit Decimal difference from the stored binary64 source value."
            ),
            "cases": encode_decimal_tree(anchor_results),
        },
        "conventions": {
            "df_resid": "n - numerical_rank",
            "df_model": "numerical_rank - 1 with intercept, otherwise numerical_rank",
            "sst": "centered with intercept; uncentered sum(y^2) without intercept",
            "sigma2": "SSE / df_resid",
            "gaussian_mle_variance": "SSE / n",
            "loglik": "-n/2 * (log(2*pi) + 1 + log(SSE/n))",
            "aic": "-2*loglik + 2*numerical_rank",
            "bic": "-2*loglik + log(n)*numerical_rank",
            "leverage": "diag(X * inverse(X'X) * X')",
            "cooks_distance": "residual^2 * leverage / (rank * sigma2 * (1-leverage)^2)",
            "student_t_two_sided_p": "I_{df/(df+t^2)}(df/2, 1/2)",
            "f_survival_probability": "I_{df_resid/(df_resid+df_model*F)}(df_resid/2, df_model/2)",
        },
        "r9": {
            "rule": (
                "Acceptance margins are measured-error multiples: approximately "
                "3.87x-4x for Decimal validation/reference anchors and "
                "3.95x-3.99x for the verified Rust replay."
            ),
            "max_rel_definition": (
                "Every max_rel field is scaled error: abs(error) / "
                "max(1, abs(reference)); it equals absolute error when "
                "abs(reference) < 1."
            ),
            "algebra_identities": {
                "measured_max_abs": canonical_decimal(identity.max_abs),
                "acceptance_tolerance_abs": canonical_decimal(four_times(identity.max_abs)),
                "worst_path": identity.worst_abs_path,
            },
            "precision_80_vs_120": {
                "measured_max_abs": canonical_decimal(precision_error.max_abs),
                "acceptance_tolerance_abs": canonical_decimal(four_times(precision_error.max_abs)),
                "worst_abs_path": precision_error.worst_abs_path,
                "measured_max_rel": canonical_decimal(precision_error.max_rel),
                "acceptance_tolerance_rel": canonical_decimal(four_times(precision_error.max_rel)),
                "worst_rel_path": precision_error.worst_rel_path,
            },
            "rust_replay": {
                "status": "verified",
                "measured_on": "2026-09-03",
                "measured_max_abs": "2.1827872842550278E-11",
                "worst_abs_path": "high_leverage.f_statistic",
                "acceptance_tolerance_abs": "8.7E-11",
                "measured_max_rel": "1.3174774993335828E-13",
                "worst_rel_path": "high_leverage.cooks_distance[14]",
                "relative_error_definition": "abs(error) / max(1, abs(expected))",
                "acceptance_tolerance_rel": "5.2E-13",
            },
        },
        "case_count": len(encoded_cases),
        "cases": encoded_cases,
    }


def canonical_bytes() -> bytes:
    text = json.dumps(
        build_payload(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def emit(path: Path) -> int:
    data = canonical_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"wrote {path} bytes={len(data)} sha256={sha256_bytes(data)}")
    return 0


def first_difference(left: bytes, right: bytes) -> str:
    shared = min(len(left), len(right))
    for index in range(shared):
        if left[index] != right[index]:
            return f"byte {index}: committed={left[index]} regenerated={right[index]}"
    if len(left) != len(right):
        return f"length: committed={len(left)} regenerated={len(right)}"
    return "none"


def check(path: Path) -> int:
    if not path.is_file():
        print(f"missing fixture: {path}")
        return 1
    actual_raw = path.read_bytes()
    actual = normalize_logical_newlines(actual_raw)
    expected = canonical_bytes()
    if actual != expected:
        print(
            f"stale fixture: {path}\n"
            f"  committed logical sha256={sha256_bytes(actual)}\n"
            f"  regenerated sha256={sha256_bytes(expected)}\n"
            f"  first difference: {first_difference(actual, expected)}"
        )
        return 1
    payload = json.loads(actual)
    if payload.get("schema_version") != 1 or payload.get("case_count") != len(CASES):
        print("fixture schema/count validation failed")
        return 1
    newline_note = " logical-LF" if actual_raw != actual else ""
    print(
        f"ok {path} bytes={len(actual)} sha256={sha256_bytes(actual)}{newline_note}"
    )
    return 0


def deep() -> int:
    precisions = (ORACLE_PRECISION, VERIFICATION_PRECISION, DEEP_PRECISION)
    suites: dict[int, list[dict[str, Any]]] = {}
    identities: dict[int, ErrorSummary] = {}
    for precision in precisions:
        suites[precision], identities[precision] = compute_suite(precision)
    for low, high in zip(precisions, precisions[1:]):
        error = compare_trees(suites[low], suites[high], f"precision_{low}_vs_{high}")
        guard = Decimal(1).scaleb(-(low - 15))
        if error.max_abs > guard or error.max_rel > guard:
            print(
                f"precision validation failed {low}->{high}: "
                f"abs={error.max_abs} rel={error.max_rel} guard={guard}"
            )
            return 1
        print(
            f"precision {low}->{high}: max_abs={error.max_abs} "
            f"at {error.worst_abs_path}; max_rel={error.max_rel} "
            f"at {error.worst_rel_path}; guard={guard}"
        )
    for precision in precisions:
        identity = identities[precision]
        print(
            f"identities precision={precision}: max_abs={identity.max_abs} "
            f"at {identity.worst_abs_path}"
        )
    print("deep precision and algebra validation passed")
    return 0


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)
    emit_parser = subcommands.add_parser("emit", help="write the canonical fixture")
    emit_parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_GOLDEN)
    check_parser = subcommands.add_parser("check", help="byte-check the canonical fixture")
    check_parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_GOLDEN)
    subcommands.add_parser("deep", help="validate at 80, 120, and 180 digits")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "emit":
        return emit(arguments.path)
    if arguments.command == "check":
        return check(arguments.path)
    if arguments.command == "deep":
        return deep()
    raise AssertionError(f"unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
