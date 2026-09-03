#!/usr/bin/env python3
"""Generate an independent Decimal brute-force oracle for generic HMMs.

The runtime uses log-space dynamic programming.  This oracle instead evaluates
every hidden-state path in ordinary high-precision Decimal probability space,
sums those path masses, and selects the largest path mass.  Gaussian pi is
computed with the Decimal Gauss--Legendre algorithm; Poisson probabilities use
an exact integer factorial.  Only the Python standard library is used.

This is deliberately not an hmmlearn fixture and does not claim hmmlearn
provenance.

Usage:
    python scripts/hmm_oracle.py emit [golden/hmm.json]
    python scripts/hmm_oracle.py check [golden/hmm.json]
    python scripts/hmm_oracle.py deep [golden/hmm.json]
"""

from __future__ import annotations

import json
import math
import sys
from decimal import Decimal, getcontext, localcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = ROOT / "golden" / "hmm.json"
EMITTED_DIGITS = 50
LOW_PRECISION = 80
HIGH_PRECISION = 120
DEEP_PRECISION = 180


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "gaussian_univariate_two_state",
        "family": "gaussian",
        "initial": ("0.65", "0.35"),
        "transition": (("0.82", "0.18"), ("0.27", "0.73")),
        "emissions": (
            {"mean": ("-1.25",), "variance": ("0.7",)},
            {"mean": ("2.0",), "variance": ("1.4",)},
        ),
        "observations": (
            ("-1.7",),
            ("-0.8",),
            ("0.4",),
            ("2.3",),
            ("1.5",),
            ("-1.1",),
        ),
        "baum_welch_iterations": 2,
    },
    {
        "name": "gaussian_diagonal_two_dim_three_state",
        "family": "gaussian",
        "initial": ("0.2", "0.5", "0.3"),
        "transition": (
            ("0.72", "0.20", "0.08"),
            ("0.15", "0.70", "0.15"),
            ("0.06", "0.24", "0.70"),
        ),
        "emissions": (
            {"mean": ("-2.0", "0.5"), "variance": ("0.8", "1.2")},
            {"mean": ("0.2", "-1.0"), "variance": ("1.1", "0.6")},
            {"mean": ("2.4", "1.8"), "variance": ("0.9", "1.5")},
        ),
        "observations": (
            ("-1.6", "0.8"),
            ("0.0", "-0.7"),
            ("2.8", "1.4"),
            ("1.9", "2.1"),
        ),
    },
    {
        "name": "categorical_three_state_with_zero_support",
        "family": "categorical",
        "initial": ("0.5", "0.3", "0.2"),
        "transition": (
            ("0.75", "0.25", "0"),
            ("0.10", "0.72", "0.18"),
            ("0", "0.22", "0.78"),
        ),
        "emissions": (
            {"weights": ("7", "2", "1", "0")},
            {"weights": ("1", "5", "3", "1")},
            {"weights": ("0", "1", "2", "7")},
        ),
        "observations": (0, 1, 3, 2, 3, 0),
    },
    {
        "name": "poisson_three_state_including_point_mass",
        "family": "poisson",
        "initial": ("0.25", "0.50", "0.25"),
        "transition": (
            ("0.68", "0.27", "0.05"),
            ("0.14", "0.72", "0.14"),
            ("0.04", "0.26", "0.70"),
        ),
        "emissions": (
            {"rate": "0"},
            {"rate": "1.2"},
            {"rate": "4.5"},
        ),
        "observations": (0, 0, 1, 4, 6, 2),
    },
)


def _decimal_pi() -> Decimal:
    """Return pi at the active Decimal context precision."""
    one = Decimal(1)
    two = Decimal(2)
    four = Decimal(4)
    a = one
    b = (one / two).sqrt()
    t = one / four
    multiplier = one
    threshold = Decimal(10) ** (-(getcontext().prec - 8))
    for _ in range(32):
        midpoint = (a + b) / two
        geometric = (a * b).sqrt()
        difference = a - midpoint
        t -= multiplier * difference * difference
        a = midpoint
        b = geometric
        multiplier *= two
        if abs(a - b) <= threshold:
            break
    else:
        raise ArithmeticError("Decimal Gauss--Legendre pi did not converge")
    return (a + b) * (a + b) / (four * t)


def _as_decimals(values: Any) -> list[Decimal]:
    return [Decimal(value) for value in values]


def _prepare_case(spec: dict[str, Any]) -> dict[str, Any]:
    family = spec["family"]
    emissions: list[dict[str, Any]] = []
    for emission in spec["emissions"]:
        if family == "gaussian":
            emissions.append(
                {
                    "mean": _as_decimals(emission["mean"]),
                    "variance": _as_decimals(emission["variance"]),
                }
            )
        elif family == "categorical":
            weights = _as_decimals(emission["weights"])
            total = sum(weights, Decimal(0))
            emissions.append(
                {
                    "weights": weights,
                    "probabilities": [weight / total for weight in weights],
                }
            )
        elif family == "poisson":
            emissions.append({"rate": Decimal(emission["rate"])})
        else:
            raise AssertionError(f"unknown family {family}")

    if family == "gaussian":
        observations: list[Any] = [
            _as_decimals(observation) for observation in spec["observations"]
        ]
    else:
        observations = list(spec["observations"])
    return {
        "family": family,
        "initial": _as_decimals(spec["initial"]),
        "transition": [_as_decimals(row) for row in spec["transition"]],
        "emissions": emissions,
        "observations": observations,
    }


def _emission_probability(
    family: str,
    emission: dict[str, Any],
    observation: Any,
    pi: Decimal,
) -> Decimal:
    if family == "gaussian":
        mean = emission["mean"]
        variance = emission["variance"]
        quadratic_and_log_det = Decimal(0)
        for value, center, scale in zip(observation, mean, variance):
            residual = value - center
            quadratic_and_log_det += scale.ln() + residual * residual / scale
        dimension = Decimal(len(mean))
        log_probability = -(
            dimension * (Decimal(2) * pi).ln() + quadratic_and_log_det
        ) / Decimal(2)
        return log_probability.exp()
    if family == "categorical":
        return emission["probabilities"][observation]
    if family == "poisson":
        rate = emission["rate"]
        if rate == 0:
            return Decimal(1) if observation == 0 else Decimal(0)
        return (
            (-rate).exp()
            * (rate**observation)
            / Decimal(math.factorial(observation))
        )
    raise AssertionError(f"unknown family {family}")


def _decode_path(encoded: int, time_count: int, state_count: int) -> list[int]:
    path: list[int] = []
    for _ in range(time_count):
        path.append(encoded % state_count)
        encoded //= state_count
    return path


def _enumerate_paths(model: dict[str, Any], pi: Decimal) -> dict[str, Any]:
    family = model["family"]
    initial = model["initial"]
    transition = model["transition"]
    emissions = model["emissions"]
    observations = model["observations"]
    time_count = len(observations)
    state_count = len(initial)
    path_count = state_count**time_count
    total = Decimal(0)
    best_mass = Decimal(-1)
    best_path: list[int] = []
    gamma_mass = [
        [Decimal(0) for _ in range(state_count)] for _ in range(time_count)
    ]
    xi_mass = [
        [
            [Decimal(0) for _ in range(state_count)]
            for _ in range(state_count)
        ]
        for _ in range(max(time_count - 1, 0))
    ]

    for encoded in range(path_count):
        path = _decode_path(encoded, time_count, state_count)
        mass = initial[path[0]] * _emission_probability(
            family, emissions[path[0]], observations[0], pi
        )
        for time in range(1, time_count):
            mass *= transition[path[time - 1]][path[time]]
            mass *= _emission_probability(
                family, emissions[path[time]], observations[time], pi
            )
        total += mass
        if mass > best_mass:
            best_mass = mass
            best_path = path
        for time, state in enumerate(path):
            gamma_mass[time][state] += mass
        for time in range(time_count - 1):
            xi_mass[time][path[time]][path[time + 1]] += mass

    if total <= 0 or best_mass <= 0:
        raise ArithmeticError("fixture has no positive-probability path")
    gamma = [[mass / total for mass in row] for row in gamma_mass]
    xi = [
        [[mass / total for mass in row] for row in matrix] for matrix in xi_mass
    ]
    return {
        "path_count": path_count,
        "sequence_probability": total,
        "log_likelihood": total.ln(),
        "gamma": gamma,
        "xi": xi,
        "viterbi_path": best_path,
        "viterbi_probability": best_mass,
        "viterbi_log_probability": best_mass.ln(),
    }


def _maximize_emissions(
    model: dict[str, Any], gamma: list[list[Decimal]]
) -> None:
    family = model["family"]
    observations = model["observations"]
    state_count = len(model["initial"])
    for state in range(state_count):
        occupancy = sum((row[state] for row in gamma), Decimal(0))
        if occupancy <= 0:
            raise ArithmeticError("Baum--Welch fixture contains an empty state")
        if family == "gaussian":
            dimension = len(observations[0])
            mean = [
                sum(
                    (
                        gamma[time][state] * observations[time][coordinate]
                        for time in range(len(observations))
                    ),
                    Decimal(0),
                )
                / occupancy
                for coordinate in range(dimension)
            ]
            variance = [
                sum(
                    (
                        gamma[time][state]
                        * (observations[time][coordinate] - mean[coordinate])
                        * (observations[time][coordinate] - mean[coordinate])
                        for time in range(len(observations))
                    ),
                    Decimal(0),
                )
                / occupancy
                for coordinate in range(dimension)
            ]
            model["emissions"][state] = {"mean": mean, "variance": variance}
        elif family == "categorical":
            categories = len(model["emissions"][state]["probabilities"])
            probabilities = []
            for category in range(categories):
                count = sum(
                    (
                        gamma[time][state]
                        for time, observation in enumerate(observations)
                        if observation == category
                    ),
                    Decimal(0),
                )
                probabilities.append(count / occupancy)
            model["emissions"][state] = {
                "weights": probabilities[:],
                "probabilities": probabilities,
            }
        elif family == "poisson":
            rate = sum(
                (
                    gamma[time][state] * Decimal(observation)
                    for time, observation in enumerate(observations)
                ),
                Decimal(0),
            ) / occupancy
            model["emissions"][state] = {"rate": rate}
        else:
            raise AssertionError(f"unknown family {family}")


def _baum_welch(
    original: dict[str, Any], iterations: int, pi: Decimal
) -> dict[str, Any]:
    model = {
        "family": original["family"],
        "initial": original["initial"][:],
        "transition": [row[:] for row in original["transition"]],
        "emissions": [
            {
                key: value[:] if isinstance(value, list) else value
                for key, value in emission.items()
            }
            for emission in original["emissions"]
        ],
        "observations": [
            observation[:] if isinstance(observation, list) else observation
            for observation in original["observations"]
        ],
    }
    for _ in range(iterations):
        posterior = _enumerate_paths(model, pi)
        gamma = posterior["gamma"]
        xi = posterior["xi"]
        _maximize_emissions(model, gamma)
        model["initial"] = gamma[0][:]
        for source in range(len(model["initial"])):
            denominator = sum(
                (gamma[time][source] for time in range(len(gamma) - 1)),
                Decimal(0),
            )
            if denominator <= 0:
                raise ArithmeticError("Baum--Welch transition state is empty")
            for destination in range(len(model["initial"])):
                numerator = sum(
                    (matrix[source][destination] for matrix in xi), Decimal(0)
                )
                model["transition"][source][destination] = numerator / denominator

    final = _enumerate_paths(model, pi)
    emissions = []
    for emission in model["emissions"]:
        if model["family"] == "gaussian":
            emissions.append(
                {"mean": emission["mean"], "variance": emission["variance"]}
            )
        elif model["family"] == "categorical":
            emissions.append({"probabilities": emission["probabilities"]})
        else:
            emissions.append({"rate": emission["rate"]})
    return {
        "iterations": iterations,
        "initial": model["initial"],
        "transition": model["transition"],
        "emissions": emissions,
        "log_likelihood_after_fit": final["log_likelihood"],
        "viterbi_path_after_fit": final["viterbi_path"],
    }


def _input_payload(spec: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    emissions = []
    for emission in model["emissions"]:
        if model["family"] == "gaussian":
            emissions.append(
                {"mean": emission["mean"], "variance": emission["variance"]}
            )
        elif model["family"] == "categorical":
            emissions.append({"weights": emission["weights"]})
        else:
            emissions.append({"rate": emission["rate"]})
    payload = {
        "initial": model["initial"],
        "transition": model["transition"],
        "emissions": emissions,
        "observations": model["observations"],
    }
    if "baum_welch_iterations" in spec:
        payload["baum_welch_iterations"] = spec["baum_welch_iterations"]
    return payload


def _solve_case(spec: dict[str, Any], precision: int) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = precision
        pi = _decimal_pi()
        model = _prepare_case(spec)
        expected = _enumerate_paths(model, pi)
        # Gamma and xi are used to derive optional Baum--Welch results but are
        # omitted from this compact fixed-parameter replay fixture.
        del expected["gamma"]
        del expected["xi"]
        result = {
            "name": spec["name"],
            "family": spec["family"],
            "input": _input_payload(spec, model),
            "expected": expected,
        }
        iterations = spec.get("baum_welch_iterations")
        if iterations is not None:
            result["baum_welch"] = _baum_welch(model, iterations, pi)
        return result


def _raw_payload(precision: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_count": len(CASES),
        "cases": [_solve_case(case, precision) for case in CASES],
    }


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
    if difference >= Decimal("1e-65"):
        raise AssertionError(f"80/120-digit disagreement {difference} at {worst_path}")
    payload = _encode(low)
    payload["metadata"] = {
        "arithmetic": "Python standard-library decimal.Decimal",
        "oracle": "ordinary-probability exhaustive hidden-path enumeration",
        "gaussian_constant": "Decimal Gauss--Legendre pi",
        "poisson_factorial": "exact Python integer factorial",
        "runtime_algorithm": "Rust log-space forward--backward and Viterbi",
        "provenance": "independent Decimal brute force; not hmmlearn",
        "low_precision_digits": LOW_PRECISION,
        "high_precision_digits": HIGH_PRECISION,
        "emitted_significant_digits": EMITTED_DIGITS,
        "max_80_120_abs_difference": format(difference, ".6e"),
        "worst_80_120_path": worst_path,
    }
    return payload


def _canonical_bytes() -> bytes:
    return (
        json.dumps(_canonical_payload(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
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
        if difference >= Decimal("1e-105"):
            raise SystemExit(
                f"120/180-digit disagreement {difference} at {worst_path}"
            )
        print(f"120/180 max abs difference {difference} at {worst_path}")
    print(f"verified {path}")


if __name__ == "__main__":
    main()
