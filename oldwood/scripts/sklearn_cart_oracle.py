#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "joblib==1.5.2",
#   "numpy==2.3.3",
#   "scikit-learn==1.7.2",
#   "scipy==1.16.2",
#   "threadpoolctl==3.6.0",
# ]
# ///
"""Generate the scikit-learn CART Tier-0 fixture.

The committed CSV stores training inputs and predictions on separate probe
rows.  It deliberately omits thresholds, child indices, and ``apply`` leaf
identifiers so equivalent tree layouts remain acceptable.

Commands::

    uv run oldwood/scripts/sklearn_cart_oracle.py emit
    uv run oldwood/scripts/sklearn_cart_oracle.py check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SCIKIT_LEARN_VERSION = "1.7.2"
SCHEMA_VERSION = 1
GENERATED_ON = "2026-09-04"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "golden" / "sklearn_cart.csv"
HEADER = (
    "case",
    "task",
    "criterion",
    "max_depth",
    "random_state",
    "role",
    "row",
    "x0",
    "x1",
    "target",
    "weight",
    "prediction",
    "class0",
    "class1",
    "prob0",
    "prob1",
)


@dataclass(frozen=True)
class Case:
    """One deterministic weighted estimator and its independent probes."""

    name: str
    task: Literal["classifier", "regressor"]
    criterion: str
    max_depth: int
    random_state: int
    train_x: tuple[tuple[float, float], ...]
    target: tuple[int | float, ...]
    weight: tuple[float, ...]
    probe_x: tuple[tuple[float, float], ...]


CASES = (
    Case(
        name="weighted_gini_depth2",
        task="classifier",
        criterion="gini",
        max_depth=2,
        random_state=1729,
        train_x=(
            (0.0, 2.0),
            (1.0, 1.5),
            (2.0, 0.0),
            (3.0, 3.0),
            (4.0, 2.5),
            (5.0, 2.0),
            (6.0, 0.5),
            (7.0, 1.0),
        ),
        target=(2, 2, 9, 9, 9, 2, 2, 9),
        weight=(1.0, 3.0, 1.0, 2.0, 4.0, 1.0, 2.0, 3.0),
        probe_x=(
            (-1.0, 2.0),
            (1.5, 1.0),
            (2.5, 0.5),
            (3.5, 2.8),
            (5.5, 1.8),
            (6.5, 0.8),
            (8.0, 2.0),
        ),
    ),
    Case(
        name="weighted_entropy_depth2",
        task="classifier",
        criterion="entropy",
        max_depth=2,
        random_state=1729,
        train_x=(
            (-3.0, 0.0),
            (-2.0, 2.0),
            (-1.0, 1.0),
            (0.0, 3.0),
            (1.0, 0.5),
            (2.0, 2.5),
            (3.0, 1.5),
            (4.0, 4.0),
        ),
        target=(3, 3, 8, 3, 8, 8, 3, 8),
        weight=(2.0, 1.0, 4.0, 1.0, 3.0, 2.0, 1.0, 3.0),
        probe_x=(
            (-4.0, 1.0),
            (-1.5, 1.5),
            (-0.5, 2.0),
            (0.5, 0.8),
            (1.5, 2.8),
            (3.5, 1.0),
            (5.0, 4.0),
        ),
    ),
    Case(
        name="weighted_squared_error_depth2",
        task="regressor",
        criterion="squared_error",
        max_depth=2,
        random_state=1729,
        train_x=(
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
            (7.0, 0.0),
        ),
        target=(0.0, 1.0, 1.5, 5.0, 7.0, 7.5, 11.0, 13.0),
        weight=(1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 2.0),
        probe_x=(
            (-1.0, 0.0),
            (0.5, 0.0),
            (2.5, 0.0),
            (3.5, 0.0),
            (4.5, 0.0),
            (6.5, 0.0),
            (8.0, 0.0),
        ),
    ),
)


def binary64(value: int | float) -> str:
    """Return a locale-independent round-trippable binary64 spelling."""

    return format(float(value), ".17g")


def train_rows(case: Case) -> list[list[str]]:
    """Fit one reference estimator and emit its train and probe rows."""

    train_x = np.asarray(case.train_x, dtype=np.float64)
    probe_x = np.asarray(case.probe_x, dtype=np.float64)
    weight = np.asarray(case.weight, dtype=np.float64)
    if case.task == "classifier":
        estimator = DecisionTreeClassifier(
            criterion=case.criterion,
            splitter="best",
            max_depth=case.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            random_state=case.random_state,
            ccp_alpha=0.0,
        ).fit(train_x, np.asarray(case.target, dtype=np.int64), sample_weight=weight)
        prediction = estimator.predict(probe_x)
        probability = estimator.predict_proba(probe_x)
        classes = tuple(int(value) for value in estimator.classes_)
        if len(classes) != 2:
            raise AssertionError(f"{case.name} must remain binary")
    else:
        estimator = DecisionTreeRegressor(
            criterion=case.criterion,
            splitter="best",
            max_depth=case.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            random_state=case.random_state,
            ccp_alpha=0.0,
        ).fit(train_x, np.asarray(case.target, dtype=np.float64), sample_weight=weight)
        prediction = estimator.predict(probe_x)
        probability = None
        classes = None

    rows: list[list[str]] = []
    shared = [
        case.name,
        case.task,
        case.criterion,
        str(case.max_depth),
        str(case.random_state),
    ]
    for index, ((x0, x1), target, row_weight) in enumerate(
        zip(case.train_x, case.target, case.weight, strict=True)
    ):
        rows.append(
            shared
            + [
                "train",
                str(index),
                binary64(x0),
                binary64(x1),
                str(target),
                binary64(row_weight),
                "",
                "",
                "",
                "",
                "",
            ]
        )
    for index, (x0, x1) in enumerate(case.probe_x):
        if case.task == "classifier":
            assert classes is not None and probability is not None
            expected = [
                str(int(prediction[index])),
                str(classes[0]),
                str(classes[1]),
                binary64(probability[index, 0]),
                binary64(probability[index, 1]),
            ]
        else:
            expected = [binary64(prediction[index]), "", "", "", ""]
        rows.append(
            shared
            + [
                "probe",
                str(index),
                binary64(x0),
                binary64(x1),
                "",
                "",
            ]
            + expected
        )
    return rows


def payload() -> str:
    """Create canonical LF-only CSV text."""

    if sklearn.__version__ != SCIKIT_LEARN_VERSION:
        raise RuntimeError(
            f"expected scikit-learn {SCIKIT_LEARN_VERSION}, found {sklearn.__version__}"
        )
    stream = io.StringIO(newline="")
    stream.write(f"# schema={SCHEMA_VERSION}\n")
    stream.write(f"# generated_on={GENERATED_ON}\n")
    stream.write(f"# scikit_learn={SCIKIT_LEARN_VERSION}\n")
    stream.write("# source=sklearn.tree.DecisionTreeClassifier,DecisionTreeRegressor\n")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(HEADER)
    for case in CASES:
        writer.writerows(train_rows(case))
    return stream.getvalue()


def normalized_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "check"))
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generated = payload()
    if arguments.command == "emit":
        arguments.path.parent.mkdir(parents=True, exist_ok=True)
        arguments.path.write_text(generated, encoding="utf-8", newline="\n")
        digest = hashlib.sha256(generated.encode()).hexdigest()
        print(f"wrote {arguments.path} sha256={digest}")
        return
    committed = normalized_text(arguments.path)
    if committed != generated:
        raise SystemExit(f"fixture differs: regenerate {arguments.path}")
    digest = hashlib.sha256(committed.encode()).hexdigest()
    print(f"fixture matches sha256={digest}")


if __name__ == "__main__":
    main()
