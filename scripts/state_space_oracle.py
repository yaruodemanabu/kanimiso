#!/usr/bin/env python3
"""Deterministic Decimal oracle for linear-Gaussian state-space models.

The authoritative values are obtained by constructing the joint Gaussian law
of every state and observation and directly conditioning on the observed
block.  A sequential Joseph-form Kalman filter and RTS smoother exists only as
an independent ``deep`` cross-check; it never supplies golden expected values.

Usage:
    python scripts/state_space_oracle.py emit golden/state_space.json
    python scripts/state_space_oracle.py check golden/state_space.json
    python scripts/state_space_oracle.py deep golden/state_space.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Sequence


D = Decimal
Vector = list[Decimal]
Matrix = list[list[Decimal]]

DEFAULT_GOLDEN = Path("golden/state_space.json")
GENERATED_ON = "2026-09-03"
ORACLE_PRECISION = 80
VERIFICATION_PRECISION = 120
DEEP_PRECISION = 180
PRECISIONS = (ORACLE_PRECISION, VERIFICATION_PRECISION, DEEP_PRECISION)
OUTPUT_SIGNIFICANT_DIGITS = 64
INTERNAL_GUARD_DIGITS = 12
CANONICAL_RUNTIME = "CPython 3.12.13; libmpdec 2.5.1"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    purpose: str
    transition_matrix: tuple[tuple[str, ...], ...]
    transition_offset: tuple[str, ...]
    process_covariance: tuple[tuple[str, ...], ...]
    observation_matrix: tuple[tuple[str, ...], ...]
    observation_offset: tuple[str, ...]
    observation_covariance: tuple[tuple[str, ...], ...]
    initial_predicted_mean: tuple[str, ...]
    initial_predicted_covariance: tuple[tuple[str, ...], ...]
    observations: tuple[tuple[str | None, ...], ...]


CASES = (
    CaseSpec(
        name="scalar_local_level",
        purpose="Scalar baseline with nonzero process and observation noise.",
        transition_matrix=(("1",),),
        transition_offset=("0",),
        process_covariance=(("0.2",),),
        observation_matrix=(("1",),),
        observation_offset=("0",),
        observation_covariance=(("0.5",),),
        initial_predicted_mean=("0",),
        initial_predicted_covariance=(("1",),),
        observations=(("1",), ("-0.5",), ("2",), ("1.5",)),
    ),
    CaseSpec(
        name="local_linear_trend_regression",
        purpose=(
            "Two-state local-linear-trend regression y=[0,2,-1,3,0]; the "
            "alternating observations expose the historical smoother indexing bug."
        ),
        transition_matrix=(("1", "1"), ("0", "1")),
        transition_offset=("0", "0"),
        process_covariance=(("0.10", "0.02"), ("0.02", "0.05")),
        observation_matrix=(("1", "0"),),
        observation_offset=("0",),
        observation_covariance=(("0.4",),),
        initial_predicted_mean=("0", "0"),
        initial_predicted_covariance=(("1", "0.2"), ("0.2", "1")),
        observations=(("0",), ("2",), ("-1",), ("3",), ("0",)),
    ),
    CaseSpec(
        name="correlated_multivariate",
        purpose=(
            "Two states and two jointly observed channels with non-diagonal Q and R, "
            "a dense H, and nonzero transition and observation offsets."
        ),
        transition_matrix=(("0.8", "0.1"), ("-0.2", "0.9")),
        transition_offset=("0.1", "-0.05"),
        process_covariance=(("0.3", "0.08"), ("0.08", "0.2")),
        observation_matrix=(("1", "0.4"), ("-0.3", "1.2")),
        observation_offset=("0.2", "-0.1"),
        observation_covariance=(("0.5", "0.12"), ("0.12", "0.4")),
        initial_predicted_mean=("0.5", "-0.2"),
        initial_predicted_covariance=(("1", "0.25"), ("0.25", "0.8")),
        observations=(
            ("1.1", "-0.4"),
            ("0.2", "0.8"),
            ("-0.5", "1.3"),
            ("0.7", "-0.2"),
        ),
    ),
    CaseSpec(
        name="partial_and_all_missing",
        purpose=(
            "Partial and fully missing rows exercise observed H/R submatrices and "
            "the predict-through-all-missing convention."
        ),
        transition_matrix=(("1", "0.5"), ("0", "0.9")),
        transition_offset=("-0.1", "0.05"),
        process_covariance=(("0.15", "0.03"), ("0.03", "0.1")),
        observation_matrix=(("1", "0"), ("0.5", "1")),
        observation_offset=("0", "0.2"),
        observation_covariance=(("0.4", "0.1"), ("0.1", "0.6")),
        initial_predicted_mean=("0", "1"),
        initial_predicted_covariance=(("0.9", "0.2"), ("0.2", "0.7")),
        observations=(
            ("1", None),
            (None, None),
            ("0.2", "-0.3"),
            (None, "0.8"),
            ("1.2", None),
        ),
    ),
)


def decimal_vector(values: Sequence[str]) -> Vector:
    return [D(value) for value in values]


def decimal_matrix(rows: Sequence[Sequence[str]]) -> Matrix:
    return [[D(value) for value in row] for row in rows]


def decimal_observations(
    rows: Sequence[Sequence[str | None]],
) -> list[list[Decimal | None]]:
    return [[None if value is None else D(value) for value in row] for row in rows]


def zeros(rows: int, columns: int) -> Matrix:
    return [[D(0) for _ in range(columns)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    return [[D(1) if row == column else D(0) for column in range(size)] for row in range(size)]


def transpose(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix, strict=True)]


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][column] + right[row][column] for column in range(len(left[row]))]
        for row in range(len(left))
    ]


def matrix_sub(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left[row][column] - right[row][column] for column in range(len(left[row]))]
        for row in range(len(left))
    ]


def matrix_scale(matrix: Matrix, scalar: Decimal) -> Matrix:
    return [[scalar * value for value in row] for row in matrix]


def matrix_mul(left: Matrix, right: Matrix) -> Matrix:
    if not left:
        return []
    if not right:
        return zeros(len(left), 0)
    inner = len(right)
    if len(left[0]) != inner:
        raise ValueError(f"matrix product shape mismatch {len(left)}x{len(left[0])} and {inner}x{len(right[0])}")
    columns = len(right[0])
    return [
        [sum((left[i][k] * right[k][j] for k in range(inner)), D(0)) for j in range(columns)]
        for i in range(len(left))
    ]


def matrix_vector_mul(matrix: Matrix, vector: Vector) -> Vector:
    return [
        sum((matrix[row][column] * vector[column] for column in range(len(vector))), D(0))
        for row in range(len(matrix))
    ]


def vector_add(left: Vector, right: Vector) -> Vector:
    return [a + b for a, b in zip(left, right, strict=True)]


def vector_sub(left: Vector, right: Vector) -> Vector:
    return [a - b for a, b in zip(left, right, strict=True)]


def symmetric(matrix: Matrix) -> Matrix:
    half = D("0.5")
    other = transpose(matrix)
    return [
        [half * (matrix[i][j] + other[i][j]) for j in range(len(matrix[i]))]
        for i in range(len(matrix))
    ]


def select_rows(matrix: Matrix, indices: Sequence[int]) -> Matrix:
    return [list(matrix[index]) for index in indices]


def select_square(matrix: Matrix, indices: Sequence[int]) -> Matrix:
    return [[matrix[row][column] for column in indices] for row in indices]


def cholesky(matrix: Matrix) -> Matrix:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("Cholesky requires a square matrix")
    lower = zeros(size, size)
    for row in range(size):
        for column in range(row + 1):
            residual = matrix[row][column] - sum(
                (lower[row][k] * lower[column][k] for k in range(column)), D(0)
            )
            if row == column:
                if residual <= 0:
                    raise ArithmeticError(f"matrix is not positive definite at diagonal {row}: {residual}")
                lower[row][column] = residual.sqrt()
            else:
                lower[row][column] = residual / lower[column][column]
    return lower


def solve_cholesky(lower: Matrix, rhs: Vector) -> Vector:
    size = len(lower)
    forward = [D(0) for _ in range(size)]
    for row in range(size):
        subtotal = sum((lower[row][k] * forward[k] for k in range(row)), D(0))
        forward[row] = (rhs[row] - subtotal) / lower[row][row]
    upper = transpose(lower)
    solution = [D(0) for _ in range(size)]
    for row in range(size - 1, -1, -1):
        subtotal = sum((upper[row][k] * solution[k] for k in range(row + 1, size)), D(0))
        solution[row] = (forward[row] - subtotal) / upper[row][row]
    return solution


def inverse_spd(matrix: Matrix) -> Matrix:
    if not matrix:
        return []
    lower = cholesky(matrix)
    size = len(matrix)
    columns = []
    for column in range(size):
        basis = [D(1) if row == column else D(0) for row in range(size)]
        columns.append(solve_cholesky(lower, basis))
    return symmetric(transpose(columns))


def log_determinant_spd(matrix: Matrix) -> Decimal:
    if not matrix:
        return D(0)
    lower = cholesky(matrix)
    return D(2) * sum((lower[index][index].ln() for index in range(len(lower))), D(0))


def decimal_pi(precision: int) -> Decimal:
    with localcontext() as context:
        context.prec = precision + 20
        context.rounding = ROUND_HALF_EVEN
        one = D(1)
        two = D(2)
        four = D(4)
        a = one
        b = one / two.sqrt()
        t = one / four
        multiplier = one
        for _ in range(16):
            next_a = (a + b) / two
            next_b = (a * b).sqrt()
            difference = a - next_a
            next_t = t - multiplier * difference * difference
            a, b, t = next_a, next_b, next_t
            multiplier *= two
        value = (a + b) * (a + b) / (four * t)
        context.prec = precision
        return +value


def gaussian_log_density(
    values: Vector,
    means: Vector,
    covariance: Matrix,
    pi: Decimal,
) -> Decimal:
    if not values:
        return D(0)
    centered = vector_sub(values, means)
    solved = solve_cholesky(cholesky(covariance), centered)
    quadratic = sum((a * b for a, b in zip(centered, solved, strict=True)), D(0))
    dimension = D(len(values))
    return -D("0.5") * (
        dimension * (D(2) * pi).ln() + log_determinant_spd(covariance) + quadratic
    )


def validate_case_dimensions(spec: CaseSpec) -> None:
    state_size = len(spec.initial_predicted_mean)
    observation_size = len(spec.observation_offset)
    square_state = (spec.transition_matrix, spec.process_covariance, spec.initial_predicted_covariance)
    for matrix in square_state:
        if len(matrix) != state_size or any(len(row) != state_size for row in matrix):
            raise ValueError(f"{spec.name}: malformed state matrix")
    if len(spec.transition_offset) != state_size:
        raise ValueError(f"{spec.name}: malformed transition offset")
    if len(spec.observation_matrix) != observation_size or any(
        len(row) != state_size for row in spec.observation_matrix
    ):
        raise ValueError(f"{spec.name}: malformed observation matrix")
    if len(spec.observation_covariance) != observation_size or any(
        len(row) != observation_size for row in spec.observation_covariance
    ):
        raise ValueError(f"{spec.name}: malformed observation covariance")
    if not spec.observations or any(len(row) != observation_size for row in spec.observations):
        raise ValueError(f"{spec.name}: malformed observations")


def materialize_case(spec: CaseSpec) -> dict[str, Any]:
    validate_case_dimensions(spec)
    return {
        "transition_matrix": decimal_matrix(spec.transition_matrix),
        "transition_offset": decimal_vector(spec.transition_offset),
        "process_covariance": decimal_matrix(spec.process_covariance),
        "observation_matrix": decimal_matrix(spec.observation_matrix),
        "observation_offset": decimal_vector(spec.observation_offset),
        "observation_covariance": decimal_matrix(spec.observation_covariance),
        "initial_predicted_mean": decimal_vector(spec.initial_predicted_mean),
        "initial_predicted_covariance": decimal_matrix(spec.initial_predicted_covariance),
        "observations": decimal_observations(spec.observations),
    }


def scalar_bilinear(left: Vector, matrix: Matrix, right: Vector) -> Decimal:
    product = matrix_vector_mul(matrix, right)
    return sum((a * b for a, b in zip(left, product, strict=True)), D(0))


def build_joint_gaussian(case: dict[str, Any]) -> dict[str, Any]:
    transition = case["transition_matrix"]
    transition_t = transpose(transition)
    offset = case["transition_offset"]
    process = case["process_covariance"]
    observation = case["observation_matrix"]
    observation_t = transpose(observation)
    observation_offset = case["observation_offset"]
    observation_covariance = case["observation_covariance"]
    initial_mean = case["initial_predicted_mean"]
    initial_covariance = case["initial_predicted_covariance"]
    time_count = len(case["observations"])
    state_size = len(initial_mean)
    observation_size = len(observation)

    state_means: list[Vector] = [list(initial_mean)]
    for _ in range(1, time_count):
        state_means.append(vector_add(matrix_vector_mul(transition, state_means[-1]), offset))

    state_covariances = [
        [zeros(state_size, state_size) for _ in range(time_count)]
        for _ in range(time_count)
    ]
    state_covariances[0][0] = symmetric(initial_covariance)
    for time in range(1, time_count):
        for earlier in range(time):
            cross = matrix_mul(transition, state_covariances[time - 1][earlier])
            state_covariances[time][earlier] = cross
            state_covariances[earlier][time] = transpose(cross)
        state_covariances[time][time] = symmetric(
            matrix_add(
                matrix_mul(matrix_mul(transition, state_covariances[time - 1][time - 1]), transition_t),
                process,
            )
        )

    observation_means = [
        vector_add(matrix_vector_mul(observation, state_means[time]), observation_offset)
        for time in range(time_count)
    ]
    observation_covariances = [
        [zeros(observation_size, observation_size) for _ in range(time_count)]
        for _ in range(time_count)
    ]
    for left_time in range(time_count):
        for right_time in range(time_count):
            block = matrix_mul(
                matrix_mul(observation, state_covariances[left_time][right_time]),
                observation_t,
            )
            if left_time == right_time:
                block = matrix_add(block, observation_covariance)
            observation_covariances[left_time][right_time] = block

    return {
        "state_means": state_means,
        "state_covariances": state_covariances,
        "observation_means": observation_means,
        "observation_covariances": observation_covariances,
    }


ObservationCoordinate = tuple[int, int]


def observed_coordinates(case: dict[str, Any]) -> list[ObservationCoordinate]:
    return [
        (time, channel)
        for time, row in enumerate(case["observations"])
        for channel, value in enumerate(row)
        if value is not None
    ]


def coordinate_values(case: dict[str, Any], coordinates: Sequence[ObservationCoordinate]) -> Vector:
    values = []
    for time, channel in coordinates:
        value = case["observations"][time][channel]
        if value is None:
            raise AssertionError("missing coordinate entered an observed block")
        values.append(value)
    return values


def coordinate_means(joint: dict[str, Any], coordinates: Sequence[ObservationCoordinate]) -> Vector:
    return [joint["observation_means"][time][channel] for time, channel in coordinates]


def coordinate_covariance(
    joint: dict[str, Any], coordinates: Sequence[ObservationCoordinate]
) -> Matrix:
    return [
        [
            joint["observation_covariances"][left_time][right_time][left_channel][right_channel]
            for right_time, right_channel in coordinates
        ]
        for left_time, left_channel in coordinates
    ]


def state_observation_cross_covariance(
    joint: dict[str, Any],
    case: dict[str, Any],
    state_time: int,
    coordinates: Sequence[ObservationCoordinate],
) -> Matrix:
    observation = case["observation_matrix"]
    state_size = len(joint["state_means"][state_time])
    cross = zeros(state_size, len(coordinates))
    for column, (observation_time, channel) in enumerate(coordinates):
        block = joint["state_covariances"][state_time][observation_time]
        for state_index in range(state_size):
            cross[state_index][column] = sum(
                (block[state_index][k] * observation[channel][k] for k in range(state_size)),
                D(0),
            )
    return cross


def condition_state(
    joint: dict[str, Any],
    case: dict[str, Any],
    state_time: int,
    coordinates: Sequence[ObservationCoordinate],
) -> tuple[Vector, Matrix]:
    mean = list(joint["state_means"][state_time])
    covariance = [list(row) for row in joint["state_covariances"][state_time][state_time]]
    if not coordinates:
        return mean, covariance
    values = coordinate_values(case, coordinates)
    observation_mean = coordinate_means(joint, coordinates)
    observation_covariance = coordinate_covariance(joint, coordinates)
    inverse = inverse_spd(observation_covariance)
    cross = state_observation_cross_covariance(joint, case, state_time, coordinates)
    centered = vector_sub(values, observation_mean)
    correction = matrix_vector_mul(matrix_mul(cross, inverse), centered)
    conditional_covariance = symmetric(
        matrix_sub(covariance, matrix_mul(matrix_mul(cross, inverse), transpose(cross)))
    )
    return vector_add(mean, correction), conditional_covariance


def prefix_log_density(
    joint: dict[str, Any],
    case: dict[str, Any],
    coordinates: Sequence[ObservationCoordinate],
    pi: Decimal,
) -> Decimal:
    if not coordinates:
        return D(0)
    return gaussian_log_density(
        coordinate_values(case, coordinates),
        coordinate_means(joint, coordinates),
        coordinate_covariance(joint, coordinates),
        pi,
    )


def output_from_conditionals(
    case: dict[str, Any],
    predicted_means: list[Vector],
    predicted_covariances: list[Matrix],
    filtered_means: list[Vector],
    filtered_covariances: list[Matrix],
    smoothed_means: list[Vector],
    smoothed_covariances: list[Matrix],
    loglik_by_time: list[Decimal],
) -> dict[str, Any]:
    observation = case["observation_matrix"]
    observation_offset = case["observation_offset"]
    observation_covariance = case["observation_covariance"]
    transition = case["transition_matrix"]
    observations = case["observations"]
    time_count = len(observations)
    state_size = len(predicted_means[0])

    innovations: list[list[Decimal | None]] = []
    full_innovation_covariances: list[Matrix] = []
    observed_indices: list[list[int]] = []
    observed_innovations: list[Vector] = []
    observed_innovation_covariances: list[Matrix] = []
    kalman_gains: list[Matrix] = []
    observed_counts: list[int] = []
    for time in range(time_count):
        expected_observation = vector_add(
            matrix_vector_mul(observation, predicted_means[time]), observation_offset
        )
        full_covariance = symmetric(
            matrix_add(
                matrix_mul(matrix_mul(observation, predicted_covariances[time]), transpose(observation)),
                observation_covariance,
            )
        )
        indices = [index for index, value in enumerate(observations[time]) if value is not None]
        innovation: list[Decimal | None] = []
        for index, value in enumerate(observations[time]):
            innovation.append(None if value is None else value - expected_observation[index])
        observed_innovation = [innovation[index] for index in indices]
        if any(value is None for value in observed_innovation):
            raise AssertionError("observed innovation unexpectedly contains null")
        observed_innovation = [value for value in observed_innovation if value is not None]
        observed_covariance = select_square(full_covariance, indices)
        if indices:
            observed_h = select_rows(observation, indices)
            gain = matrix_mul(
                matrix_mul(predicted_covariances[time], transpose(observed_h)),
                inverse_spd(observed_covariance),
            )
        else:
            gain = zeros(state_size, 0)
        innovations.append(innovation)
        full_innovation_covariances.append(full_covariance)
        observed_indices.append(indices)
        observed_innovations.append(observed_innovation)
        observed_innovation_covariances.append(observed_covariance)
        kalman_gains.append(gain)
        observed_counts.append(len(indices))

    rts_gains = [
        matrix_mul(
            matrix_mul(filtered_covariances[time], transpose(transition)),
            inverse_spd(predicted_covariances[time + 1]),
        )
        for time in range(time_count - 1)
    ]
    return {
        "predicted_means": predicted_means,
        "predicted_covariances": predicted_covariances,
        "filtered_means": filtered_means,
        "filtered_covariances": filtered_covariances,
        "smoothed_means": smoothed_means,
        "smoothed_covariances": smoothed_covariances,
        "innovations": innovations,
        "full_innovation_covariances": full_innovation_covariances,
        "observed_indices": observed_indices,
        "observed_innovations": observed_innovations,
        "observed_innovation_covariances": observed_innovation_covariances,
        "kalman_gains": kalman_gains,
        "rts_gains": rts_gains,
        "loglik_by_time": loglik_by_time,
        "total_loglik": sum(loglik_by_time, D(0)),
        "observed_counts": observed_counts,
    }


def authoritative_case(spec: CaseSpec, precision: int) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_HALF_EVEN
        case = materialize_case(spec)
        joint = build_joint_gaussian(case)
        coordinates = observed_coordinates(case)
        time_count = len(case["observations"])
        pi = decimal_pi(precision)

        predicted_means: list[Vector] = []
        predicted_covariances: list[Matrix] = []
        filtered_means: list[Vector] = []
        filtered_covariances: list[Matrix] = []
        smoothed_means: list[Vector] = []
        smoothed_covariances: list[Matrix] = []
        loglik_by_time: list[Decimal] = []
        previous_prefix_loglik = D(0)

        for time in range(time_count):
            past = [coordinate for coordinate in coordinates if coordinate[0] < time]
            current = [coordinate for coordinate in coordinates if coordinate[0] <= time]
            predicted_mean, predicted_covariance = condition_state(joint, case, time, past)
            filtered_mean, filtered_covariance = condition_state(joint, case, time, current)
            smoothed_mean, smoothed_covariance = condition_state(joint, case, time, coordinates)
            current_prefix_loglik = prefix_log_density(joint, case, current, pi)

            predicted_means.append(predicted_mean)
            predicted_covariances.append(predicted_covariance)
            filtered_means.append(filtered_mean)
            filtered_covariances.append(filtered_covariance)
            smoothed_means.append(smoothed_mean)
            smoothed_covariances.append(smoothed_covariance)
            loglik_by_time.append(current_prefix_loglik - previous_prefix_loglik)
            previous_prefix_loglik = current_prefix_loglik

        result = output_from_conditionals(
            case,
            predicted_means,
            predicted_covariances,
            filtered_means,
            filtered_covariances,
            smoothed_means,
            smoothed_covariances,
            loglik_by_time,
        )
        result["total_loglik"] = previous_prefix_loglik
        return result


def sequential_joseph_case(spec: CaseSpec, precision: int) -> dict[str, Any]:
    """Secondary implementation used only by ``deep``."""

    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_HALF_EVEN
        case = materialize_case(spec)
        transition = case["transition_matrix"]
        transition_offset = case["transition_offset"]
        process_covariance = case["process_covariance"]
        observation = case["observation_matrix"]
        observation_covariance = case["observation_covariance"]
        observations = case["observations"]
        state_size = len(case["initial_predicted_mean"])
        pi = decimal_pi(precision)

        predicted_means: list[Vector] = []
        predicted_covariances: list[Matrix] = []
        filtered_means: list[Vector] = []
        filtered_covariances: list[Matrix] = []
        loglik_by_time: list[Decimal] = []
        predicted_mean = list(case["initial_predicted_mean"])
        predicted_covariance = [list(row) for row in case["initial_predicted_covariance"]]

        for time, row in enumerate(observations):
            predicted_means.append(predicted_mean)
            predicted_covariances.append(predicted_covariance)
            indices = [index for index, value in enumerate(row) if value is not None]
            if indices:
                observed_h = select_rows(observation, indices)
                observed_r = select_square(observation_covariance, indices)
                observed_offset = [case["observation_offset"][index] for index in indices]
                observed_values = [row[index] for index in indices]
                if any(value is None for value in observed_values):
                    raise AssertionError("selected observations contain null")
                values = [value for value in observed_values if value is not None]
                expected = vector_add(matrix_vector_mul(observed_h, predicted_mean), observed_offset)
                innovation = vector_sub(values, expected)
                innovation_covariance = symmetric(
                    matrix_add(
                        matrix_mul(matrix_mul(observed_h, predicted_covariance), transpose(observed_h)),
                        observed_r,
                    )
                )
                gain = matrix_mul(
                    matrix_mul(predicted_covariance, transpose(observed_h)),
                    inverse_spd(innovation_covariance),
                )
                filtered_mean = vector_add(predicted_mean, matrix_vector_mul(gain, innovation))
                joseph_left = matrix_sub(identity(state_size), matrix_mul(gain, observed_h))
                filtered_covariance = symmetric(
                    matrix_add(
                        matrix_mul(matrix_mul(joseph_left, predicted_covariance), transpose(joseph_left)),
                        matrix_mul(matrix_mul(gain, observed_r), transpose(gain)),
                    )
                )
                loglik = gaussian_log_density(innovation, [D(0)] * len(indices), innovation_covariance, pi)
            else:
                filtered_mean = list(predicted_mean)
                filtered_covariance = [list(values) for values in predicted_covariance]
                loglik = D(0)
            filtered_means.append(filtered_mean)
            filtered_covariances.append(filtered_covariance)
            loglik_by_time.append(loglik)
            if time + 1 < len(observations):
                predicted_mean = vector_add(matrix_vector_mul(transition, filtered_mean), transition_offset)
                predicted_covariance = symmetric(
                    matrix_add(
                        matrix_mul(matrix_mul(transition, filtered_covariance), transpose(transition)),
                        process_covariance,
                    )
                )

        smoothed_means = [list(values) for values in filtered_means]
        smoothed_covariances = [[list(row) for row in matrix] for matrix in filtered_covariances]
        for time in range(len(observations) - 2, -1, -1):
            gain = matrix_mul(
                matrix_mul(filtered_covariances[time], transpose(transition)),
                inverse_spd(predicted_covariances[time + 1]),
            )
            smoothed_means[time] = vector_add(
                filtered_means[time],
                matrix_vector_mul(gain, vector_sub(smoothed_means[time + 1], predicted_means[time + 1])),
            )
            smoothed_covariances[time] = symmetric(
                matrix_add(
                    filtered_covariances[time],
                    matrix_mul(
                        matrix_mul(
                            gain,
                            matrix_sub(smoothed_covariances[time + 1], predicted_covariances[time + 1]),
                        ),
                        transpose(gain),
                    ),
                )
            )

        return output_from_conditionals(
            case,
            predicted_means,
            predicted_covariances,
            filtered_means,
            filtered_covariances,
            smoothed_means,
            smoothed_covariances,
            loglik_by_time,
        )


@dataclass
class ErrorSummary:
    max_abs: Decimal = D(0)
    worst_abs_path: str = "none"
    max_rel: Decimal = D(0)
    worst_rel_path: str = "none"

    def observe(self, actual: Decimal, expected: Decimal, path: str) -> None:
        with localcontext() as context:
            context.prec = DEEP_PRECISION + 30
            difference = abs(actual - expected)
            relative = difference / max(D(1), abs(expected))
        if difference > self.max_abs:
            self.max_abs = difference
            self.worst_abs_path = path
        if relative > self.max_rel:
            self.max_rel = relative
            self.worst_rel_path = path

    def merge(self, other: "ErrorSummary") -> None:
        if other.max_abs > self.max_abs:
            self.max_abs = other.max_abs
            self.worst_abs_path = other.worst_abs_path
        if other.max_rel > self.max_rel:
            self.max_rel = other.max_rel
            self.worst_rel_path = other.worst_rel_path


def compare_trees(left: Any, right: Any, path: str = "root") -> ErrorSummary:
    summary = ErrorSummary()

    def walk(actual: Any, expected: Any, current: str) -> None:
        if isinstance(actual, Decimal) and isinstance(expected, Decimal):
            summary.observe(actual, expected, current)
            return
        if type(actual) is not type(expected):
            raise TypeError(
                f"{current}: type mismatch {type(actual).__name__} != {type(expected).__name__}"
            )
        if isinstance(actual, dict):
            if set(actual) != set(expected):
                raise KeyError(f"{current}: key mismatch {sorted(actual)} != {sorted(expected)}")
            for key in sorted(actual):
                walk(actual[key], expected[key], f"{current}.{key}")
            return
        if isinstance(actual, (list, tuple)):
            if len(actual) != len(expected):
                raise ValueError(f"{current}: length mismatch {len(actual)} != {len(expected)}")
            for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
                walk(actual_item, expected_item, f"{current}[{index}]")
            return
        if actual != expected:
            raise ValueError(f"{current}: semantic mismatch {actual!r} != {expected!r}")

    walk(left, right, path)
    return summary


def compare_schema(left: Any, right: Any, path: str = "root") -> None:
    if type(left) is not type(right):
        raise TypeError(f"{path}: schema type mismatch {type(left).__name__} != {type(right).__name__}")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise KeyError(f"{path}: schema keys differ {sorted(left)} != {sorted(right)}")
        for key in sorted(left):
            compare_schema(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError(f"{path}: schema list length {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            compare_schema(left_item, right_item, f"{path}[{index}]")


def algebra_identity_summary(spec: CaseSpec, result: dict[str, Any], precision: int) -> ErrorSummary:
    with localcontext() as context:
        context.prec = precision
        context.rounding = ROUND_HALF_EVEN
        case = materialize_case(spec)
        transition = case["transition_matrix"]
        process_covariance = case["process_covariance"]
        observation = case["observation_matrix"]
        observation_covariance = case["observation_covariance"]
        observations = case["observations"]
        state_size = len(case["initial_predicted_mean"])
        summary = ErrorSummary()

        summary.merge(
            compare_trees(
                result["predicted_means"][0],
                case["initial_predicted_mean"],
                f"{spec.name}.initial_predicted_mean",
            )
        )
        summary.merge(
            compare_trees(
                result["predicted_covariances"][0],
                case["initial_predicted_covariance"],
                f"{spec.name}.initial_predicted_covariance",
            )
        )

        for time in range(len(observations)):
            predicted_mean = result["predicted_means"][time]
            predicted_covariance = result["predicted_covariances"][time]
            filtered_mean = result["filtered_means"][time]
            filtered_covariance = result["filtered_covariances"][time]
            indices = result["observed_indices"][time]
            actual_indices = [
                index for index, value in enumerate(observations[time]) if value is not None
            ]
            if indices != actual_indices:
                raise AssertionError(
                    f"{spec.name}: observed indices {indices} != {actual_indices} at {time}"
                )
            observed_h = select_rows(observation, indices)
            observed_r = select_square(observation_covariance, indices)
            gain = result["kalman_gains"][time]
            innovation = result["observed_innovations"][time]
            innovation_covariance = result["observed_innovation_covariances"][time]
            full_covariance = symmetric(
                matrix_add(
                    matrix_mul(matrix_mul(observation, predicted_covariance), transpose(observation)),
                    observation_covariance,
                )
            )
            summary.merge(
                compare_trees(
                    result["full_innovation_covariances"][time],
                    full_covariance,
                    f"{spec.name}.full_innovation_covariance[{time}]",
                )
            )
            summary.merge(
                compare_trees(
                    innovation_covariance,
                    select_square(full_covariance, indices),
                    f"{spec.name}.observed_innovation_covariance[{time}]",
                )
            )
            expected_observation = vector_add(
                matrix_vector_mul(observation, predicted_mean), case["observation_offset"]
            )
            expected_full_innovation = [
                None if value is None else value - expected_observation[index]
                for index, value in enumerate(observations[time])
            ]
            summary.merge(
                compare_trees(
                    result["innovations"][time],
                    expected_full_innovation,
                    f"{spec.name}.innovation_identity[{time}]",
                )
            )
            summary.merge(
                compare_trees(
                    innovation,
                    [expected_full_innovation[index] for index in indices],
                    f"{spec.name}.observed_innovation_identity[{time}]",
                )
            )
            if result["observed_counts"][time] != len(indices):
                raise AssertionError(f"{spec.name}: observed count mismatch at {time}")
            if indices:
                summary.merge(
                    compare_trees(
                        matrix_mul(gain, innovation_covariance),
                        matrix_mul(predicted_covariance, transpose(observed_h)),
                        f"{spec.name}.kalman_gain_identity[{time}]",
                    )
                )
                summary.merge(
                    compare_trees(
                        filtered_mean,
                        vector_add(predicted_mean, matrix_vector_mul(gain, innovation)),
                        f"{spec.name}.filtered_mean_identity[{time}]",
                    )
                )
                joseph_left = matrix_sub(identity(state_size), matrix_mul(gain, observed_h))
                joseph = symmetric(
                    matrix_add(
                        matrix_mul(matrix_mul(joseph_left, predicted_covariance), transpose(joseph_left)),
                        matrix_mul(matrix_mul(gain, observed_r), transpose(gain)),
                    )
                )
                summary.merge(
                    compare_trees(
                        filtered_covariance,
                        joseph,
                        f"{spec.name}.joseph_covariance_identity[{time}]",
                    )
                )
            else:
                summary.merge(
                    compare_trees(
                        filtered_mean,
                        predicted_mean,
                        f"{spec.name}.all_missing_mean_identity[{time}]",
                    )
                )
                summary.merge(
                    compare_trees(
                        filtered_covariance,
                        predicted_covariance,
                        f"{spec.name}.all_missing_covariance_identity[{time}]",
                    )
                )
                summary.observe(result["loglik_by_time"][time], D(0), f"{spec.name}.all_missing_loglik[{time}]")

            for label in ("predicted_covariances", "filtered_covariances", "smoothed_covariances"):
                covariance = result[label][time]
                cholesky(covariance)
                summary.merge(
                    compare_trees(covariance, transpose(covariance), f"{spec.name}.{label}_symmetry[{time}]")
                )

            if time + 1 < len(observations):
                summary.merge(
                    compare_trees(
                        result["predicted_means"][time + 1],
                        vector_add(matrix_vector_mul(transition, filtered_mean), case["transition_offset"]),
                        f"{spec.name}.predict_mean_identity[{time}]",
                    )
                )
                predicted_next = symmetric(
                    matrix_add(
                        matrix_mul(matrix_mul(transition, filtered_covariance), transpose(transition)),
                        process_covariance,
                    )
                )
                summary.merge(
                    compare_trees(
                        result["predicted_covariances"][time + 1],
                        predicted_next,
                        f"{spec.name}.predict_covariance_identity[{time}]",
                    )
                )
                smoother_gain = result["rts_gains"][time]
                summary.merge(
                    compare_trees(
                        matrix_mul(smoother_gain, result["predicted_covariances"][time + 1]),
                        matrix_mul(filtered_covariance, transpose(transition)),
                        f"{spec.name}.rts_gain_identity[{time}]",
                    )
                )
                summary.merge(
                    compare_trees(
                        result["smoothed_means"][time],
                        vector_add(
                            filtered_mean,
                            matrix_vector_mul(
                                smoother_gain,
                                vector_sub(
                                    result["smoothed_means"][time + 1],
                                    result["predicted_means"][time + 1],
                                ),
                            ),
                        ),
                        f"{spec.name}.rts_mean_identity[{time}]",
                    )
                )
                summary.merge(
                    compare_trees(
                        result["smoothed_covariances"][time],
                        symmetric(
                            matrix_add(
                                filtered_covariance,
                                matrix_mul(
                                    matrix_mul(
                                        smoother_gain,
                                        matrix_sub(
                                            result["smoothed_covariances"][time + 1],
                                            result["predicted_covariances"][time + 1],
                                        ),
                                    ),
                                    transpose(smoother_gain),
                                ),
                            )
                        ),
                        f"{spec.name}.rts_covariance_identity[{time}]",
                    )
                )

        summary.merge(
            compare_trees(
                result["smoothed_means"][-1],
                result["filtered_means"][-1],
                f"{spec.name}.terminal_smoothed_mean",
            )
        )
        summary.merge(
            compare_trees(
                result["smoothed_covariances"][-1],
                result["filtered_covariances"][-1],
                f"{spec.name}.terminal_smoothed_covariance",
            )
        )
        summary.observe(
            result["total_loglik"],
            sum(result["loglik_by_time"], D(0)),
            f"{spec.name}.loglik_sum",
        )
        guard = D(1).scaleb(-(precision - INTERNAL_GUARD_DIGITS))
        if summary.max_abs > guard:
            raise AssertionError(
                f"{spec.name}: algebra identity error {summary.max_abs} at "
                f"{summary.worst_abs_path} exceeds precision-derived guard {guard}"
            )
        return summary


def authoritative_suite(precision: int) -> tuple[list[dict[str, Any]], ErrorSummary]:
    results = []
    identities = ErrorSummary()
    for spec in CASES:
        result = authoritative_case(spec, precision)
        identities.merge(algebra_identity_summary(spec, result, precision))
        results.append(result)
    return results, identities


def canonical_decimal(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError(f"non-finite Decimal cannot enter the fixture: {value}")
    if value == 0:
        return "0"
    with localcontext() as context:
        context.prec = OUTPUT_SIGNIFICANT_DIGITS
        context.rounding = ROUND_HALF_EVEN
        return format((+value).normalize(), "E")


def encode_decimal_tree(value: Any) -> Any:
    if isinstance(value, Decimal):
        return canonical_decimal(value)
    if isinstance(value, dict):
        return {key: encode_decimal_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_decimal_tree(item) for item in value]
    return value


def precision_guard(precision: int) -> Decimal:
    return D(1).scaleb(-(precision - INTERNAL_GUARD_DIGITS))


def assert_output_stable(left: Any, right: Any, label: str) -> None:
    encoded_left = encode_decimal_tree(left)
    encoded_right = encode_decimal_tree(right)
    if encoded_left != encoded_right:
        difference = first_tree_difference(encoded_left, encoded_right)
        raise AssertionError(
            f"{label}: {OUTPUT_SIGNIFICANT_DIGITS}-digit output is not stable; {difference}"
        )


def first_tree_difference(left: Any, right: Any, path: str = "root") -> str:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in sorted(left):
            difference = first_tree_difference(left[key], right[key], f"{path}.{key}")
            if difference != "none":
                return difference
        return "none"
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = first_tree_difference(left_item, right_item, f"{path}[{index}]")
            if difference != "none":
                return difference
        return "none"
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return "none"


def normalize_logical_newlines(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def script_sha256() -> str:
    source = normalize_logical_newlines(Path(__file__).resolve().read_bytes())
    return hashlib.sha256(source).hexdigest()


def validation_payload(
    suites: dict[int, list[dict[str, Any]]],
    identities: dict[int, ErrorSummary],
) -> dict[str, Any]:
    precision_results: dict[str, Any] = {}
    for lower, higher in zip(PRECISIONS[:-1], PRECISIONS[1:], strict=True):
        label = f"{lower}_to_{higher}"
        error = compare_trees(suites[lower], suites[higher], f"precision_{label}")
        guard = precision_guard(lower)
        if error.max_abs > guard or error.max_rel > guard:
            raise AssertionError(
                f"precision {label} unstable: abs={error.max_abs} rel={error.max_rel} guard={guard}"
            )
        assert_output_stable(suites[lower], suites[higher], f"precision {label}")
        precision_results[label] = {
            "measured_max_abs": canonical_decimal(error.max_abs),
            "worst_abs_path": error.worst_abs_path,
            "measured_max_rel": canonical_decimal(error.max_rel),
            "worst_rel_path": error.worst_rel_path,
            "precision_derived_guard": canonical_decimal(guard),
            "encoded_output_identical": True,
        }
    return {
        "precision_stability": precision_results,
        "algebra_identities": {
            str(precision): {
                "measured_max_abs": canonical_decimal(identities[precision].max_abs),
                "worst_abs_path": identities[precision].worst_abs_path,
                "measured_max_rel": canonical_decimal(identities[precision].max_rel),
                "worst_rel_path": identities[precision].worst_rel_path,
                "precision_derived_guard": canonical_decimal(precision_guard(precision)),
            }
            for precision in PRECISIONS
        },
        "output_stability_rule": (
            f"Decimal trees encoded at {OUTPUT_SIGNIFICANT_DIGITS} significant digits must be "
            "exactly identical for 80->120 and 120->180 digit recomputation."
        ),
        "sequential_joseph_rts_crosscheck": {
            "status": "deep_only",
            "method": (
                "A separate sequential Joseph-form Kalman filter and RTS recursion is compared "
                "against the authoritative block-conditioned results at 80, 120, and 180 digits."
            ),
        },
    }


def build_payload() -> dict[str, Any]:
    suites: dict[int, list[dict[str, Any]]] = {}
    identities: dict[int, ErrorSummary] = {}
    for precision in PRECISIONS:
        suites[precision], identities[precision] = authoritative_suite(precision)

    cases = []
    for spec, result in zip(CASES, suites[ORACLE_PRECISION], strict=True):
        materialized = materialize_case(spec)
        cases.append(
            {
                "name": spec.name,
                "purpose": spec.purpose,
                "input": encode_decimal_tree(materialized),
                "expected": encode_decimal_tree(result),
            }
        )

    return {
        "schema_version": 1,
        "generated_on": GENERATED_ON,
        "generator": {
            "path": "scripts/state_space_oracle.py",
            "sha256": script_sha256(),
            "sha256_canonicalization": "logical source bytes with CRLF and CR normalized to LF",
            "commands": {
                "emit": "python scripts/state_space_oracle.py emit golden/state_space.json",
                "check": "python scripts/state_space_oracle.py check golden/state_space.json",
                "deep": "python scripts/state_space_oracle.py deep golden/state_space.json",
            },
        },
        "oracle": {
            "authoritative_method": (
                "Construct the full joint Gaussian mean/covariance of x_0..x_{T-1} and "
                "y_0..y_{T-1}, then directly condition x_t on observed y blocks."
            ),
            "runtime_used_for_canonical_fixture": CANONICAL_RUNTIME,
            "external_dependencies": [],
            "arithmetic": "Python standard-library decimal.Decimal with ROUND_HALF_EVEN",
            "matrix_solver": "Decimal Cholesky factorization and triangular solves",
            "pi": "Decimal Gauss-Legendre iteration",
            "canonical_precision_digits": ORACLE_PRECISION,
            "verification_precision_digits": VERIFICATION_PRECISION,
            "deep_precision_digits": DEEP_PRECISION,
            "output_significant_digits": OUTPUT_SIGNIFICANT_DIGITS,
        },
        "provenance": {
            "derivation": "Joint multivariate-normal block conditioning; no Kalman recurrence in expected-value generation.",
            "filter_reference": "Kalman linear-Gaussian conditioning equations",
            "smoother_reference": "Rauch-Tung-Striebel fixed-interval smoothing equations",
            "fixture_design": "Hand-authored deterministic decimal inputs; no random generator.",
        },
        "contract": {
            "parameter_time_dependence": "F, c, Q, H, d, and R are constant across time",
            "initial_state": "initial mean/covariance are the t=0 pre-observation prediction",
            "transition": "after updating t, predict t+1 with F*x_t + c and F*P_t*F' + Q",
            "filter_covariance_crosscheck": "sequential deep check uses Joseph covariance",
            "smoother_gain": "J_t = P_filt_t * F' * inverse(P_pred_{t+1})",
            "missing_values": "JSON null; partial updates use the observed H rows and R principal submatrix",
            "all_missing_time": "update is identity and loglik increment is zero, then transition still runs",
            "innovation": "full-length vector with null at missing channels",
            "full_innovation_covariance": "H*P_pred_t*H' + R for every channel, including missing channels",
            "loglik": (
                "observed components only; each increment is log density of the observed prefix "
                "through t minus the prefix through t-1"
            ),
        },
        "validation": validation_payload(suites, identities),
        "rust_replay": {
            "status": "verified",
            "measured_max_abs": "3.55271367880050093e-15",
            "worst_abs_path": "local_linear_trend_regression.total_loglik",
            "acceptance_tolerance_abs": "1.5e-14",
            "measured_max_rel": "8.32667268468867405e-16",
            "worst_rel_path": "local_linear_trend_regression.predicted_means[3][0]",
            "acceptance_tolerance_rel": "3.4e-15",
            "policy": "Tolerances are approximately four times the measured Rust replay maxima on 2026-09-03.",
        },
        "case_count": len(cases),
        "cases": cases,
    }


def canonical_bytes() -> bytes:
    rendered = json.dumps(
        build_payload(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (rendered + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def first_difference(left: bytes, right: bytes) -> str:
    shared = min(len(left), len(right))
    for index in range(shared):
        if left[index] != right[index]:
            return f"byte {index}: committed={left[index]} regenerated={right[index]}"
    if len(left) != len(right):
        return f"length: committed={len(left)} regenerated={len(right)}"
    return "none"


def emit(path: Path) -> int:
    data = canonical_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"wrote {path} bytes={len(data)} sha256={sha256_bytes(data)}")
    return 0


def check(path: Path) -> int:
    if not path.is_file():
        print(f"missing fixture: {path}")
        return 1
    actual_raw = path.read_bytes()
    actual = normalize_logical_newlines(actual_raw)
    expected = canonical_bytes()
    try:
        actual_document = json.loads(actual.decode("utf-8"))
        expected_document = json.loads(expected.decode("utf-8"))
        compare_schema(actual_document, expected_document)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
        print(f"fixture schema validation failed: {error}")
        return 1
    if actual != expected:
        print(
            f"stale fixture: {path}\n"
            f"  committed logical sha256={sha256_bytes(actual)}\n"
            f"  regenerated sha256={sha256_bytes(expected)}\n"
            f"  first difference: {first_difference(actual, expected)}"
        )
        return 1
    newline_note = " logical-LF" if actual_raw != actual else ""
    print(f"ok {path} bytes={len(actual)} sha256={sha256_bytes(actual)}{newline_note}")
    return 0


def deep(path: Path) -> int:
    suites: dict[int, list[dict[str, Any]]] = {}
    for precision in PRECISIONS:
        suites[precision], identities = authoritative_suite(precision)
        sequential = [sequential_joseph_case(spec, precision) for spec in CASES]
        crosscheck = compare_trees(
            suites[precision], sequential, f"block_vs_sequential_{precision}"
        )
        guard = precision_guard(precision)
        if crosscheck.max_abs > guard or crosscheck.max_rel > guard:
            print(
                f"sequential cross-check failed precision={precision}: "
                f"abs={crosscheck.max_abs} rel={crosscheck.max_rel} guard={guard}"
            )
            return 1
        assert_output_stable(
            suites[precision], sequential, f"block vs sequential precision={precision}"
        )
        print(
            f"precision={precision} identities max_abs={identities.max_abs} "
            f"at {identities.worst_abs_path}; block-vs-sequential max_abs={crosscheck.max_abs} "
            f"at {crosscheck.worst_abs_path}; max_rel={crosscheck.max_rel} "
            f"at {crosscheck.worst_rel_path}; guard={guard}"
        )

    for lower, higher in zip(PRECISIONS[:-1], PRECISIONS[1:], strict=True):
        error = compare_trees(suites[lower], suites[higher], f"precision_{lower}_to_{higher}")
        guard = precision_guard(lower)
        if error.max_abs > guard or error.max_rel > guard:
            print(
                f"precision validation failed {lower}->{higher}: "
                f"abs={error.max_abs} rel={error.max_rel} guard={guard}"
            )
            return 1
        assert_output_stable(suites[lower], suites[higher], f"precision {lower}->{higher}")
        print(
            f"precision {lower}->{higher}: max_abs={error.max_abs} at {error.worst_abs_path}; "
            f"max_rel={error.max_rel} at {error.worst_rel_path}; guard={guard}; "
            f"encoded_{OUTPUT_SIGNIFICANT_DIGITS}_digits=identical"
        )
    if check(path) != 0:
        return 1
    print("deep joint-conditioning, algebra, Joseph-filter, and RTS validation passed")
    return 0


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subcommands = command.add_subparsers(dest="command", required=True)
    emit_parser = subcommands.add_parser("emit", help="write the canonical fixture")
    emit_parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_GOLDEN)
    check_parser = subcommands.add_parser("check", help="byte-check the canonical fixture")
    check_parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_GOLDEN)
    deep_parser = subcommands.add_parser("deep", help="validate at 80, 120, and 180 digits")
    deep_parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_GOLDEN)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "emit":
        return emit(arguments.path)
    if arguments.command == "check":
        return check(arguments.path)
    if arguments.command == "deep":
        return deep(arguments.path)
    raise AssertionError(f"unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
