#!/usr/bin/env python3
"""Generate and verify the online-RLS Decimal oracle.

The oracle does not replay the recursive Kalman-gain update used by Rust.
Instead, it solves the equivalent geometrically weighted batch normal
equations, including the finite initial covariance prior.  Only Python's
standard-library ``decimal`` module is used.

Usage:
    python scripts/online_rls_oracle.py emit [golden/online_rls.json]
    python scripts/online_rls_oracle.py check [golden/online_rls.json]
    python scripts/online_rls_oracle.py deep [golden/online_rls.json]
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "golden" / "online_rls.json"
EMITTED_DIGITS = 48
LOW_PRECISION = 80
HIGH_PRECISION = 120
DEEP_PRECISION = 180


CASES = (
    {
        "name": "one_feature_no_intercept_growing_window",
        "forgetting_factor": "1",
        "p0": "1000",
        "fit_intercept": False,
        "x": (("1",), ("2",), ("-1",), ("0.5",), ("3",)),
        "y": ("2.2", "4.1", "-2.1", "1.2", "5.9"),
        "prediction_x": (("-2",), ("0",), ("4",)),
    },
    {
        "name": "two_features_intercept_forgetting",
        "forgetting_factor": "0.93",
        "p0": "25",
        "fit_intercept": True,
        "x": (
            ("1.0", "-0.5"),
            ("2.0", "1.0"),
            ("-1.0", "2.0"),
            ("0.25", "-1.5"),
            ("3.0", "0.75"),
            ("-2.0", "-0.25"),
            ("1.5", "1.25"),
        ),
        "y": ("3.75", "3.0", "-3.0", "3.125", "4.625", "-1.375", "2.875"),
        "prediction_x": (("0", "0"), ("2", "-1"), ("-0.5", "3")),
    },
    {
        "name": "near_unity_forgetting_stable_effective_sample",
        "forgetting_factor": "0.999999999999",
        "p0": "4",
        "fit_intercept": True,
        "x": (("-3",), ("-1",), ("0",), ("1",), ("2",), ("4",)),
        "y": ("-4.9", "-1.8", "0.4", "2.7", "4.6", "8.5"),
        "prediction_x": (("-10",), ("0.125",), ("12",)),
    },
)


def _inverse(matrix: list[list[Decimal]]) -> list[list[Decimal]]:
    n = len(matrix)
    augmented = [
        row[:] + [Decimal(int(i == j)) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for column in range(n):
        pivot_row = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        pivot = augmented[pivot_row][column]
        if pivot == 0:
            raise ArithmeticError("oracle normal matrix is singular")
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
    return [row[n:] for row in augmented]


def _matvec(matrix: list[list[Decimal]], vector: list[Decimal]) -> list[Decimal]:
    return [sum((a * b for a, b in zip(row, vector)), Decimal(0)) for row in matrix]


def _solve_case(spec: dict[str, Any], precision: int) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = precision
        forgetting = Decimal(spec["forgetting_factor"])
        p0 = Decimal(spec["p0"])
        x = [[Decimal(value) for value in row] for row in spec["x"]]
        y = [Decimal(value) for value in spec["y"]]
        prediction_x = [
            [Decimal(value) for value in row] for row in spec["prediction_x"]
        ]
        fit_intercept = bool(spec["fit_intercept"])
        design = [([Decimal(1)] + row if fit_intercept else row[:]) for row in x]
        prediction_design = [
            ([Decimal(1)] + row if fit_intercept else row[:])
            for row in prediction_x
        ]
        n = len(design)
        p = len(design[0])
        normal = [
            [
                (forgetting**n / p0 if row == column else Decimal(0))
                for column in range(p)
            ]
            for row in range(p)
        ]
        rhs = [Decimal(0) for _ in range(p)]
        for index, (features, target) in enumerate(zip(design, y)):
            weight = forgetting ** (n - 1 - index)
            for row in range(p):
                rhs[row] += weight * features[row] * target
                for column in range(p):
                    normal[row][column] += weight * features[row] * features[column]
        covariance = _inverse(normal)
        theta = _matvec(covariance, rhs)
        predictions = [
            sum((a * b for a, b in zip(row, theta)), Decimal(0))
            for row in prediction_design
        ]
        effective_sample_size = sum(
            (forgetting**power for power in range(n)), Decimal(0)
        )
        residual = max(
            abs(value - expected)
            for value, expected in zip(_matvec(normal, theta), rhs)
        )
        inverse_residual = max(
            abs(
                sum(
                    (normal[row][k] * covariance[k][column] for k in range(p)),
                    Decimal(0),
                )
                - Decimal(int(row == column))
            )
            for row in range(p)
            for column in range(p)
        )
        return {
            "name": spec["name"],
            "forgetting_factor": forgetting,
            "p0": p0,
            "fit_intercept": fit_intercept,
            "x": x,
            "y": y,
            "prediction_x": prediction_x,
            "theta": theta,
            "inverse_gram": covariance,
            "predictions": predictions,
            "effective_sample_size": effective_sample_size,
            "normal_equation_max_abs_residual": residual,
            "inverse_identity_max_abs_residual": inverse_residual,
        }


def _raw_payload(precision: int) -> dict[str, Any]:
    return {"cases": [_solve_case(case, precision) for case in CASES]}


def _compare(left: Any, right: Any, path: str = "$") -> tuple[Decimal, str]:
    if type(left) is not type(right):
        raise AssertionError(f"type mismatch at {path}: {type(left)} != {type(right)}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"key mismatch at {path}")
        worst = (Decimal(0), path)
        for key in sorted(left):
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
    max_difference, worst_path = _compare(low, high)
    if max_difference >= Decimal("1e-70"):
        raise AssertionError(
            f"80/120-digit oracle disagreement {max_difference} at {worst_path}"
        )
    encoded = _encode(low)
    encoded["metadata"] = {
        "arithmetic": "Python stdlib Decimal",
        "oracle": "geometrically weighted batch normal equations",
        "low_precision_digits": LOW_PRECISION,
        "high_precision_digits": HIGH_PRECISION,
        "emitted_significant_digits": EMITTED_DIGITS,
        "max_80_120_abs_difference": format(max_difference, ".6e"),
        "worst_80_120_path": worst_path,
    }
    return encoded


def _canonical_bytes() -> bytes:
    return (
        json.dumps(_canonical_payload(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _path(argument: str | None) -> Path:
    return Path(argument).resolve() if argument else DEFAULT_GOLDEN


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"emit", "check", "deep"}:
        raise SystemExit("expected: emit | check | deep [golden path]")
    mode = sys.argv[1]
    path = _path(sys.argv[2] if len(sys.argv) > 2 else None)
    if mode == "emit":
        payload = _canonical_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"wrote {path} ({len(payload)} bytes)")
        return
    if not path.exists():
        raise SystemExit(f"missing golden: {path}")
    expected = _canonical_bytes()
    actual = path.read_bytes()
    if actual != expected:
        raise SystemExit(f"stale or non-canonical golden: {path}")
    if mode == "deep":
        high = _raw_payload(HIGH_PRECISION)
        deep = _raw_payload(DEEP_PRECISION)
        difference, worst_path = _compare(high, deep)
        if difference >= Decimal("1e-110"):
            raise SystemExit(
                f"120/180-digit disagreement {difference} at {worst_path}"
            )
        print(f"120/180 max abs difference {difference} at {worst_path}")
    print(f"verified {path}")


if __name__ == "__main__":
    main()
