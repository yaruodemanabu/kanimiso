#!/usr/bin/env python3
"""Generate an independent Decimal state-space oracle for ARMA covariance.

Rust solves finite Yule--Walker equations.  This oracle instead builds a
companion state containing lagged observations and innovations, solves the
discrete Lyapunov equation by Decimal Gaussian elimination, and propagates
cross-covariances with powers of the transition matrix.  It uses only the
Python standard library.

Usage:
    python scripts/arma_acov_oracle.py emit [golden/arma_acov.json]
    python scripts/arma_acov_oracle.py check [golden/arma_acov.json]
    python scripts/arma_acov_oracle.py deep [golden/arma_acov.json]
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "golden" / "arma_acov.json"
EMITTED_DIGITS = 48
LOW_PRECISION = 80
HIGH_PRECISION = 120
DEEP_PRECISION = 180

CASES = (
    {"name": "white_noise", "ar": (), "ma": (), "nlags": 5},
    {"name": "ma_two", "ar": (), "ma": ("0.4", "-0.3"), "nlags": 6},
    {"name": "near_unit_ar_one", "ar": ("0.999",), "ma": (), "nlags": 12},
    {"name": "arma_one_one", "ar": ("0.7",), "ma": ("-0.2",), "nlags": 10},
    {
        "name": "stable_ar_two_ma_one",
        "ar": ("1.5", "-0.75"),
        "ma": ("0.2",),
        "nlags": 12,
    },
    {
        "name": "mixed_arma_two_two",
        "ar": ("0.55", "-0.18"),
        "ma": ("0.35", "0.12"),
        "nlags": 12,
    },
)


def _zeros(rows: int, columns: int) -> list[list[Decimal]]:
    return [[Decimal(0) for _ in range(columns)] for _ in range(rows)]


def _matmul(
    left: list[list[Decimal]], right: list[list[Decimal]]
) -> list[list[Decimal]]:
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    return [
        [
            sum((left[i][k] * right[k][j] for k in range(inner)), Decimal(0))
            for j in range(columns)
        ]
        for i in range(rows)
    ]


def _solve(matrix: list[list[Decimal]], rhs: list[Decimal]) -> list[Decimal]:
    n = len(matrix)
    augmented = [matrix[row][:] + [rhs[row]] for row in range(n)]
    for column in range(n):
        pivot_row = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if augmented[pivot_row][column] == 0:
            raise ArithmeticError("singular Decimal Lyapunov system")
        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(n)]


def _solve_case(spec: dict[str, Any], precision: int) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = precision
        ar = [Decimal(value) for value in spec["ar"]]
        ma = [Decimal(value) for value in spec["ma"]]
        p = len(ar)
        q = len(ma)
        y_dimension = max(p, 1)
        dimension = y_dimension + q

        transition = _zeros(dimension, dimension)
        for index, coefficient in enumerate(ar):
            transition[0][index] = coefficient
        for index, coefficient in enumerate(ma):
            transition[0][y_dimension + index] = coefficient
        for index in range(1, y_dimension):
            transition[index][index - 1] = Decimal(1)
        for index in range(1, q):
            transition[y_dimension + index][y_dimension + index - 1] = Decimal(1)

        loading = [Decimal(0) for _ in range(dimension)]
        loading[0] = Decimal(1)
        if q:
            loading[y_dimension] = Decimal(1)

        size = dimension * dimension
        lyapunov = _zeros(size, size)
        noise = [Decimal(0) for _ in range(size)]
        for row in range(dimension):
            for column in range(dimension):
                equation = row * dimension + column
                lyapunov[equation][equation] = Decimal(1)
                noise[equation] = loading[row] * loading[column]
                for left in range(dimension):
                    for right in range(dimension):
                        unknown = left * dimension + right
                        lyapunov[equation][unknown] -= (
                            transition[row][left] * transition[column][right]
                        )

        covariance_vector = _solve(lyapunov, noise)
        covariance = [
            covariance_vector[row * dimension : (row + 1) * dimension]
            for row in range(dimension)
        ]
        propagated = [row[:] for row in covariance]
        autocovariance: list[Decimal] = []
        for _ in range(spec["nlags"] + 1):
            autocovariance.append(propagated[0][0])
            propagated = _matmul(transition, propagated)
        autocorrelation = [value / autocovariance[0] for value in autocovariance]

        lyapunov_residual = Decimal(0)
        for row in range(size):
            residual = (
                sum(
                    (lyapunov[row][column] * covariance_vector[column] for column in range(size)),
                    Decimal(0),
                )
                - noise[row]
            )
            lyapunov_residual = max(lyapunov_residual, abs(residual))
        recurrence_residual = Decimal(0)
        for lag in range(max(p, q + 1), len(autocovariance)):
            expected = sum(
                (ar[order - 1] * autocovariance[lag - order] for order in range(1, p + 1)),
                Decimal(0),
            )
            recurrence_residual = max(
                recurrence_residual, abs(autocovariance[lag] - expected)
            )

        return {
            "name": spec["name"],
            "input": {
                "ar": ar,
                "ma": ma,
                "nlags": spec["nlags"],
                "innovation_variance": Decimal(1),
            },
            "autocovariance": autocovariance,
            "autocorrelation": autocorrelation,
            "lyapunov_max_abs_residual": lyapunov_residual,
            "recurrence_max_abs_residual": recurrence_residual,
        }


def _raw_payload(precision: int) -> dict[str, Any]:
    return {"schema_version": 1, "cases": [_solve_case(case, precision) for case in CASES]}


def _compare(left: Any, right: Any, path: str = "$") -> tuple[Decimal, str]:
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"key mismatch at {path}")
        worst = (Decimal(0), path)
        for key in left:
            candidate = _compare(left[key], right[key], f"{path}.{key}")
            if candidate[0] > worst[0]:
                worst = candidate
        return worst
    if isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"length mismatch at {path}")
        worst = (Decimal(0), path)
        for index, (a, b) in enumerate(zip(left, right)):
            candidate = _compare(a, b, f"{path}[{index}]")
            if candidate[0] > worst[0]:
                worst = candidate
        return worst
    if isinstance(left, Decimal):
        return (abs(left - right), path)
    if left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")
    return (Decimal(0), path)


def _encode(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, f".{EMITTED_DIGITS}g")
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _canonical_payload() -> dict[str, Any]:
    low = _raw_payload(LOW_PRECISION)
    high = _raw_payload(HIGH_PRECISION)
    difference, worst_path = _compare(low, high)
    if difference >= Decimal("1e-70"):
        raise AssertionError(f"80/120-digit disagreement {difference} at {worst_path}")
    payload = _encode(low)
    payload["metadata"] = {
        "arithmetic": "Python standard-library Decimal",
        "oracle": "companion-state discrete Lyapunov equation",
        "rust_algorithm": "finite ARMA Yule--Walker system",
        "low_precision_digits": LOW_PRECISION,
        "high_precision_digits": HIGH_PRECISION,
        "emitted_significant_digits": EMITTED_DIGITS,
        "max_80_120_abs_difference": format(difference, ".6e"),
        "worst_80_120_path": worst_path,
    }
    return payload


def _canonical_bytes() -> bytes:
    return (
        json.dumps(_canonical_payload(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"emit", "check", "deep"}:
        raise SystemExit("expected: emit | check | deep [golden path]")
    mode = sys.argv[1]
    path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_GOLDEN
    if mode == "emit":
        payload = _canonical_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"wrote {path} ({len(payload)} bytes)")
        return
    if not path.exists():
        raise SystemExit(f"missing golden: {path}")
    if path.read_bytes() != _canonical_bytes():
        raise SystemExit(f"stale or non-canonical golden: {path}")
    if mode == "deep":
        high = _raw_payload(HIGH_PRECISION)
        deep = _raw_payload(DEEP_PRECISION)
        difference, worst_path = _compare(high, deep)
        if difference >= Decimal("1e-110"):
            raise SystemExit(f"120/180-digit disagreement {difference} at {worst_path}")
        print(f"120/180 max abs difference {difference} at {worst_path}")
    print(f"verified {path}")


if __name__ == "__main__":
    main()
