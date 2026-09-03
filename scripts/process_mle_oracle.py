#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = []
# ///
"""Generate and verify the exponential-covariance profile-MLE oracle.

This is an independent, standard-library-only oracle for ``stats::process_mle``.
It forms the dense correlation matrix

    C[i,j] = exp(-abs(t[i] - t[j]) / rho)

and uses a Decimal Cholesky factorization for its log determinant and all GLS
solves.  In particular, it does not use the first-order Markov/OU identities
used by the production Rust implementation.

For each range on the fixed grid, the profiled quantities are

    mean   = (1' C^-1 y) / (1' C^-1 1)
    q      = (y - mean)' C^-1 (y - mean)
    sigma2 = q / n
    score  = log(det(C)) + n * log(sigma2)

where ``score`` is minus twice the Gaussian profile log likelihood with the
range-independent constant ``n * (1 + log(2*pi))`` omitted.

Usage:
    python scripts/process_mle_oracle.py emit [golden/process_mle.json]
    python scripts/process_mle_oracle.py check [golden/process_mle.json]
    python scripts/process_mle_oracle.py deep [golden/process_mle.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Sequence


D = Decimal
Vector = list[Decimal]
Matrix = list[list[Decimal]]

DEFAULT_GOLDEN = Path("golden/process_mle.json")
SCHEMA_VERSION = 1
ORACLE_PRECISION = 80
VERIFICATION_PRECISION = 120
DEEP_PRECISION = 180
OUTPUT_SIGNIFICANT_DIGITS = 56
LOW_HIGH_ABSOLUTE_TOLERANCE = D("1e-70")
HIGH_DEEP_ABSOLUTE_TOLERANCE = D("1e-110")
ENCODING_RELATIVE_TOLERANCE = D("1e-54")
ENCODING_ABSOLUTE_TOLERANCE = D("1e-54")
RANGE_MULTIPLIERS = (D(1), D(2), D(4), D(8), D(16))
# Updated only after reviewing the bytes printed by ``emit``.
EXPECTED_ARTIFACT_SHA256 = (
    "ea19073c03422e279f9cef69087750386cc951a2d91578ecbc9b97eb50a66d71"
)

ZERO = D(0)
ONE = D(1)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    purpose: str
    times: tuple[str, ...]
    observations: tuple[str, ...]


CASES = (
    CaseSpec(
        name="smooth_irregular_trend",
        purpose=(
            "Smooth observations on an irregular grid exercise the large-range "
            "end of the discrete profile."
        ),
        times=("0", "0.35", "1.10", "1.85", "3.40", "5.05", "7.20"),
        observations=("1.15", "1.24", "1.52", "1.77", "2.18", "2.61", "3.08"),
    ),
    CaseSpec(
        name="rough_irregular_signal",
        purpose=(
            "Alternating observations favor short correlation and make omission "
            "of the correlation log determinant visible."
        ),
        times=("0", "0.42", "1.30", "2.75", "3.05", "5.60", "7.15", "9.80"),
        observations=("1.2", "-0.9", "1.7", "-1.4", "0.8", "-1.8", "1.1", "-0.35"),
    ),
    CaseSpec(
        name="unsorted_clustered_times",
        purpose=(
            "A deliberately permuted irregular grid verifies that dense GLS is "
            "permutation invariant while the range grid uses sorted time gaps."
        ),
        times=("4.20", "0", "1.55", "0.20", "6.10", "3.05", "8.40"),
        observations=("0.72", "-0.40", "0.18", "-0.22", "1.05", "0.51", "1.34"),
    ),
)


def _dot(left: Sequence[Decimal], right: Sequence[Decimal]) -> Decimal:
    if len(left) != len(right):
        raise ValueError("dot-product length mismatch")
    return sum((a * b for a, b in zip(left, right, strict=True)), ZERO)


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix-vector shape mismatch")
    return [_dot(row, vector) for row in matrix]


def _cholesky(matrix: Matrix) -> Matrix:
    """Return a lower Decimal Cholesky factor of a dense SPD matrix."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Cholesky requires a nonempty square matrix")
    lower = [[ZERO for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for column in range(row + 1):
            remainder = matrix[row][column] - sum(
                (lower[row][k] * lower[column][k] for k in range(column)), ZERO
            )
            if row == column:
                if remainder <= ZERO:
                    raise ArithmeticError(
                        f"correlation matrix is not positive definite at {row}"
                    )
                lower[row][column] = remainder.sqrt()
            else:
                lower[row][column] = remainder / lower[column][column]
    return lower


def _solve_from_cholesky(lower: Matrix, rhs: Vector) -> Vector:
    n = len(lower)
    if len(rhs) != n:
        raise ValueError("Cholesky solve length mismatch")
    forward = [ZERO for _ in range(n)]
    for row in range(n):
        forward[row] = (
            rhs[row]
            - sum((lower[row][k] * forward[k] for k in range(row)), ZERO)
        ) / lower[row][row]
    solution = [ZERO for _ in range(n)]
    for row in range(n - 1, -1, -1):
        solution[row] = (
            forward[row]
            - sum(
                (lower[k][row] * solution[k] for k in range(row + 1, n)), ZERO
            )
        ) / lower[row][row]
    return solution


def _matrix_max_abs_difference(left: Matrix, right: Matrix) -> Decimal:
    return max(
        (
            abs(a - b)
            for left_row, right_row in zip(left, right, strict=True)
            for a, b in zip(left_row, right_row, strict=True)
        ),
        default=ZERO,
    )


def _cholesky_product(lower: Matrix) -> Matrix:
    n = len(lower)
    return [
        [
            sum(
                (
                    lower[row][k] * lower[column][k]
                    for k in range(min(row, column) + 1)
                ),
                ZERO,
            )
            for column in range(n)
        ]
        for row in range(n)
    ]


def _validate_spec(spec: CaseSpec) -> None:
    if not spec.name or not spec.purpose:
        raise ValueError("case names and purposes must be nonempty")
    if len(spec.times) != len(spec.observations) or len(spec.times) < 3:
        raise ValueError(f"{spec.name}: times/observations must have equal length >= 3")
    times = [D(value) for value in spec.times]
    observations = [D(value) for value in spec.observations]
    if not all(value.is_finite() for value in times + observations):
        raise ValueError(f"{spec.name}: all inputs must be finite")
    if len(set(times)) != len(times):
        raise ValueError(f"{spec.name}: observation times must be distinct")
    if min(observations) == max(observations):
        raise ValueError(f"{spec.name}: observations must be nonconstant")


def _range_grid(times: Vector) -> tuple[Decimal, tuple[Decimal, ...]]:
    ordered = sorted(times)
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    if not gaps or any(gap <= ZERO for gap in gaps):
        raise ValueError("range grid requires at least two distinct finite times")
    gaps.sort()
    # Preserve the production grid's upper-median convention for an even count.
    median_gap = gaps[len(gaps) // 2]
    return median_gap, tuple(median_gap * factor for factor in RANGE_MULTIPLIERS)


def _correlation(times: Vector, correlation_range: Decimal) -> Matrix:
    if correlation_range <= ZERO:
        raise ValueError("correlation range must be positive")
    return [
        [(-(abs(left - right) / correlation_range)).exp() for right in times]
        for left in times
    ]


def _evaluate_range(
    times: Vector, observations: Vector, correlation_range: Decimal
) -> dict[str, Decimal]:
    correlation = _correlation(times, correlation_range)
    lower = _cholesky(correlation)
    ones = [ONE for _ in times]
    inverse_ones = _solve_from_cholesky(lower, ones)
    inverse_y = _solve_from_cholesky(lower, observations)
    denominator = _dot(ones, inverse_ones)
    if denominator <= ZERO:
        raise ArithmeticError("GLS intercept denominator is not positive")
    mean = _dot(ones, inverse_y) / denominator
    residuals = [value - mean for value in observations]
    inverse_residuals = _solve_from_cholesky(lower, residuals)
    quadratic = _dot(residuals, inverse_residuals)
    if quadratic <= ZERO:
        raise ArithmeticError("profile variance is not positive")
    n_decimal = D(len(times))
    sigma2 = quadratic / n_decimal
    log_determinant = D(2) * sum((lower[i][i].ln() for i in range(len(times))), ZERO)
    profile_objective = log_determinant + n_decimal * sigma2.ln()

    solve_residual = max(
        max(
            abs(actual - expected)
            for actual, expected in zip(
                _matvec(correlation, solution), rhs, strict=True
            )
        )
        for solution, rhs in (
            (inverse_ones, ones),
            (inverse_y, observations),
            (inverse_residuals, residuals),
        )
    )
    cholesky_residual = _matrix_max_abs_difference(
        correlation, _cholesky_product(lower)
    )
    gls_orthogonality = abs(_dot(ones, inverse_residuals))
    return {
        "range": correlation_range,
        "mean": mean,
        "sigma2": sigma2,
        "quadratic_form": quadratic,
        "log_determinant": log_determinant,
        "profile_objective": profile_objective,
        "cholesky_max_abs_residual": cholesky_residual,
        "solve_max_abs_residual": solve_residual,
        "gls_orthogonality_abs": gls_orthogonality,
    }


def _solve_case(spec: CaseSpec, precision: int) -> dict[str, Any]:
    _validate_spec(spec)
    with localcontext() as context:
        context.prec = precision
        times = [D(value) for value in spec.times]
        observations = [D(value) for value in spec.observations]
        median_gap, ranges = _range_grid(times)
        evaluations = [
            _evaluate_range(times, observations, correlation_range)
            for correlation_range in ranges
        ]
        selected_index = min(
            range(len(evaluations)),
            key=lambda index: evaluations[index]["profile_objective"],
        )
        selected = evaluations[selected_index]
        ordered_scores = sorted(item["profile_objective"] for item in evaluations)
        objective_margin = ordered_scores[1] - ordered_scores[0]
        if objective_margin <= D("1e-20"):
            raise ArithmeticError(
                f"{spec.name}: range selection is not separated ({objective_margin})"
            )
        return {
            "name": spec.name,
            "purpose": spec.purpose,
            "times": times,
            "observations": observations,
            "median_positive_gap": median_gap,
            "range_grid": list(ranges),
            "evaluations": evaluations,
            "selected_index": selected_index,
            "selected": dict(selected),
            "objective_margin_to_runner_up": objective_margin,
        }


def _raw_cases(precision: int) -> list[dict[str, Any]]:
    return [_solve_case(spec, precision) for spec in CASES]


def _compare(left: Any, right: Any, path: str = "$") -> tuple[Decimal, str]:
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}: {type(left)} != {type(right)}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"key mismatch at {path}")
        worst = (ZERO, path)
        for key in sorted(left):
            candidate = _compare(left[key], right[key], f"{path}.{key}")
            if candidate[0] > worst[0]:
                worst = candidate
        return worst
    if isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        worst = (ZERO, path)
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            candidate = _compare(a, b, f"{path}[{index}]")
            if candidate[0] > worst[0]:
                worst = candidate
        return worst
    if isinstance(left, Decimal):
        return abs(left - right), path
    if left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")
    return ZERO, path


def _encode(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, f".{OUTPUT_SIGNIFICANT_DIGITS}g")
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _validate_encoded_against_raw(encoded: Any, raw: Any, path: str = "$") -> None:
    if isinstance(raw, Decimal):
        if not isinstance(encoded, str):
            raise AssertionError(f"expected decimal string at {path}")
        parsed = D(encoded)
        if not parsed.is_finite():
            raise AssertionError(f"nonfinite decimal string at {path}")
        tolerance = max(
            ENCODING_ABSOLUTE_TOLERANCE,
            abs(raw) * ENCODING_RELATIVE_TOLERANCE,
        )
        if abs(parsed - raw) > tolerance:
            raise AssertionError(
                f"encoded value differs from high precision at {path}: "
                f"{abs(parsed - raw)} > {tolerance}"
            )
        return
    if isinstance(raw, dict):
        if not isinstance(encoded, dict) or encoded.keys() != raw.keys():
            raise AssertionError(f"encoded mapping mismatch at {path}")
        for key in raw:
            _validate_encoded_against_raw(encoded[key], raw[key], f"{path}.{key}")
        return
    if isinstance(raw, list):
        if not isinstance(encoded, list) or len(encoded) != len(raw):
            raise AssertionError(f"encoded list mismatch at {path}")
        for index, (item, expected) in enumerate(zip(encoded, raw, strict=True)):
            _validate_encoded_against_raw(item, expected, f"{path}[{index}]")
        return
    if type(encoded) is not type(raw) or encoded != raw:
        raise AssertionError(f"encoded scalar mismatch at {path}")


def _require_decimal_string(value: Any, path: str) -> Decimal:
    if not isinstance(value, str):
        raise AssertionError(f"expected decimal string at {path}")
    try:
        parsed = D(value)
    except Exception as error:
        raise AssertionError(f"invalid decimal string at {path}") from error
    if not parsed.is_finite():
        raise AssertionError(f"nonfinite decimal string at {path}")
    return parsed


def _validate_schema(payload: dict[str, Any]) -> None:
    if set(payload) != {"schema_version", "metadata", "cases"}:
        raise AssertionError("root schema keys differ")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise AssertionError("schema version differs")
    metadata = payload["metadata"]
    required_metadata = {
        "arithmetic",
        "covariance",
        "factorization",
        "profile_objective",
        "range_grid",
        "oracle_precision_digits",
        "verification_precision_digits",
        "emitted_significant_digits",
        "max_80_120_abs_difference",
        "worst_80_120_path",
    }
    if not isinstance(metadata, dict) or set(metadata) != required_metadata:
        raise AssertionError("metadata schema differs")
    if payload["cases"] is None or len(payload["cases"]) != len(CASES):
        raise AssertionError("case count differs")
    case_keys = {
        "name",
        "purpose",
        "times",
        "observations",
        "median_positive_gap",
        "range_grid",
        "evaluations",
        "selected_index",
        "selected",
        "objective_margin_to_runner_up",
    }
    evaluation_keys = {
        "range",
        "mean",
        "sigma2",
        "quadratic_form",
        "log_determinant",
        "profile_objective",
        "cholesky_max_abs_residual",
        "solve_max_abs_residual",
        "gls_orthogonality_abs",
    }
    seen_names: set[str] = set()
    for case_index, case in enumerate(payload["cases"]):
        prefix = f"$.cases[{case_index}]"
        if not isinstance(case, dict) or set(case) != case_keys:
            raise AssertionError(f"case schema differs at {prefix}")
        if not isinstance(case["name"], str) or case["name"] in seen_names:
            raise AssertionError(f"case name invalid at {prefix}")
        seen_names.add(case["name"])
        if not isinstance(case["purpose"], str) or not case["purpose"]:
            raise AssertionError(f"case purpose invalid at {prefix}")
        times = case["times"]
        observations = case["observations"]
        if (
            not isinstance(times, list)
            or not isinstance(observations, list)
            or len(times) != len(observations)
            or len(times) < 3
        ):
            raise AssertionError(f"input vectors invalid at {prefix}")
        parsed_times = [
            _require_decimal_string(value, f"{prefix}.times[{index}]")
            for index, value in enumerate(times)
        ]
        parsed_observations = [
            _require_decimal_string(value, f"{prefix}.observations[{index}]")
            for index, value in enumerate(observations)
        ]
        if len(set(parsed_times)) != len(parsed_times):
            raise AssertionError(f"duplicate time at {prefix}")
        if min(parsed_observations) == max(parsed_observations):
            raise AssertionError(f"constant observations at {prefix}")
        _require_decimal_string(case["median_positive_gap"], f"{prefix}.median_positive_gap")
        if not isinstance(case["range_grid"], list) or len(case["range_grid"]) != 5:
            raise AssertionError(f"range grid invalid at {prefix}")
        ranges = [
            _require_decimal_string(value, f"{prefix}.range_grid[{index}]")
            for index, value in enumerate(case["range_grid"])
        ]
        if any(value <= ZERO for value in ranges):
            raise AssertionError(f"nonpositive range at {prefix}")
        evaluations = case["evaluations"]
        if not isinstance(evaluations, list) or len(evaluations) != 5:
            raise AssertionError(f"evaluation grid invalid at {prefix}")
        for evaluation_index, evaluation in enumerate(evaluations):
            item_path = f"{prefix}.evaluations[{evaluation_index}]"
            if not isinstance(evaluation, dict) or set(evaluation) != evaluation_keys:
                raise AssertionError(f"evaluation schema differs at {item_path}")
            for key, value in evaluation.items():
                _require_decimal_string(value, f"{item_path}.{key}")
            if D(evaluation["range"]) != ranges[evaluation_index]:
                raise AssertionError(f"evaluation range mismatch at {item_path}")
            if D(evaluation["sigma2"]) <= ZERO or D(evaluation["quadratic_form"]) <= ZERO:
                raise AssertionError(f"nonpositive profile scale at {item_path}")
        selected_index = case["selected_index"]
        if type(selected_index) is not int or not 0 <= selected_index < 5:
            raise AssertionError(f"selected index invalid at {prefix}")
        selected = case["selected"]
        if not isinstance(selected, dict) or set(selected) != evaluation_keys:
            raise AssertionError(f"selected schema differs at {prefix}")
        if selected != evaluations[selected_index]:
            raise AssertionError(f"selected evaluation mismatch at {prefix}")
        scores = [D(item["profile_objective"]) for item in evaluations]
        if selected_index != min(range(5), key=lambda index: scores[index]):
            raise AssertionError(f"selected objective is not minimal at {prefix}")
        if _require_decimal_string(
            case["objective_margin_to_runner_up"],
            f"{prefix}.objective_margin_to_runner_up",
        ) <= ZERO:
            raise AssertionError(f"objective selection is tied at {prefix}")


def _canonical_payload() -> dict[str, Any]:
    low = _raw_cases(ORACLE_PRECISION)
    high = _raw_cases(VERIFICATION_PRECISION)
    max_difference, worst_path = _compare(low, high, "$.cases")
    if max_difference >= LOW_HIGH_ABSOLUTE_TOLERANCE:
        raise AssertionError(
            f"80/120-digit disagreement {max_difference} at {worst_path}"
        )
    encoded_cases = _encode(low)
    _validate_encoded_against_raw(encoded_cases, high, "$.cases")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "arithmetic": "Python standard-library Decimal",
            "covariance": "C[i,j] = exp(-abs(t[i]-t[j])/range), without a nugget",
            "factorization": "independent dense lower Cholesky",
            "profile_objective": (
                "log(det(C)) + n*log((e' C^-1 e)/n); omits n*(1+log(2*pi))"
            ),
            "range_grid": "upper median sorted positive gap * [1,2,4,8,16]",
            "oracle_precision_digits": ORACLE_PRECISION,
            "verification_precision_digits": VERIFICATION_PRECISION,
            "emitted_significant_digits": OUTPUT_SIGNIFICANT_DIGITS,
            "max_80_120_abs_difference": format(max_difference, ".6e"),
            "worst_80_120_path": worst_path,
        },
        "cases": encoded_cases,
    }
    _validate_schema(payload)
    return payload


def _canonical_bytes() -> bytes:
    return (
        json.dumps(_canonical_payload(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _artifact_path(argument: str | None) -> Path:
    return Path(argument) if argument else DEFAULT_GOLDEN


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("emit", "check", "deep"))
    parser.add_argument("path", nargs="?", default=None)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    path = _artifact_path(arguments.path)
    expected = _canonical_bytes()
    digest = _sha256(expected)
    if arguments.mode == "emit":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
        print(f"wrote {path} ({len(expected)} bytes, sha256={digest})")
        return

    if not path.exists():
        raise SystemExit(f"missing golden: {path}")
    actual = path.read_bytes()
    if actual != expected:
        raise SystemExit(f"stale or non-canonical golden: {path}")
    parsed = json.loads(actual.decode("utf-8"))
    _validate_schema(parsed)
    if EXPECTED_ARTIFACT_SHA256 == "__UPDATE_AFTER_EMIT__":
        raise SystemExit("EXPECTED_ARTIFACT_SHA256 has not been pinned")
    if digest != EXPECTED_ARTIFACT_SHA256:
        raise SystemExit(
            f"artifact digest differs: {digest} != {EXPECTED_ARTIFACT_SHA256}"
        )

    if arguments.mode == "deep":
        high = _raw_cases(VERIFICATION_PRECISION)
        deep = _raw_cases(DEEP_PRECISION)
        difference, worst_path = _compare(high, deep, "$.cases")
        if difference >= HIGH_DEEP_ABSOLUTE_TOLERANCE:
            raise SystemExit(
                f"120/180-digit disagreement {difference} at {worst_path}"
            )
        print(f"120/180 max abs difference {difference} at {worst_path}")
    print(f"verified {path} (sha256={digest})")


if __name__ == "__main__":
    main()
