#!/usr/bin/env python3
"""Fail on Clippy regressions while the v0.1 warning backlog is removed."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys


# Measured with Rust 1.98.0 on 2026-09-04 after deduplicating the same source
# diagnostic emitted for both the library and its test harness. These budgets
# may only decrease; a previously unseen lint therefore starts with budget 0.
MAX_WARNINGS_BY_CODE = {
    "clippy::result_large_err": 311,
    "clippy::needless_range_loop": 116,
    "clippy::single_match": 18,
    "clippy::manual_contains": 13,
    "clippy::question_mark": 11,
    "clippy::too_many_arguments": 3,
    "clippy::field_reassign_with_default": 6,
    "clippy::derivable_impls": 5,
    "clippy::manual_range_contains": 3,
    "clippy::len_zero": 2,
    "clippy::bool_assert_comparison": 1,
    "clippy::collapsible_if": 1,
    "clippy::collapsible_match": 1,
    "clippy::doc_lazy_continuation": 1,
    "clippy::excessive_precision": 1,
    "clippy::manual_clamp": 1,
    "clippy::manual_div_ceil": 1,
    "clippy::unnecessary_lazy_evaluations": 1,
    "clippy::useless_vec": 1,
}
MAX_CLIPPY_WARNINGS = sum(MAX_WARNINGS_BY_CODE.values())

ROOT = Path(__file__).resolve().parents[1]
COMMAND = [
    "cargo",
    "clippy",
    "--workspace",
    "--all-targets",
    "--all-features",
    "--locked",
    "--message-format=json",
]


def main() -> int:
    env = os.environ.copy()
    env["CARGO_TERM_COLOR"] = "never"
    completed = subprocess.run(
        COMMAND,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    warning_keys: set[tuple[object, ...]] = set()
    errors: list[str] = []
    malformed: list[str] = []
    build_succeeded: bool | None = None

    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if line.strip():
                malformed.append(line)
            continue
        if event.get("reason") == "build-finished":
            build_succeeded = bool(event.get("success"))
            continue
        if event.get("reason") != "compiler-message":
            continue
        message = event.get("message", {})
        level = message.get("level")
        code = (message.get("code") or {}).get("code") or "uncoded"
        if level == "warning":
            primary_spans = tuple(
                (
                    str(span.get("file_name", "")).replace("\\", "/"),
                    span.get("line_start"),
                    span.get("column_start"),
                    span.get("line_end"),
                    span.get("column_end"),
                )
                for span in message.get("spans", [])
                if span.get("is_primary")
            )
            warning_keys.add((code, message.get("message", ""), primary_spans))
        elif level == "error":
            errors.append(message.get("rendered") or message.get("message") or code)

    if malformed:
        print("FAIL clippy ratchet: non-JSON compiler output made the count unreliable", file=sys.stderr)
        for line in malformed[:10]:
            print(line, file=sys.stderr)
        return 1

    if completed.returncode != 0 or errors or build_succeeded is not True:
        print("FAIL clippy ratchet: Clippy reported a compiler or deny-level error", file=sys.stderr)
        for error in errors[:20]:
            print(error.rstrip(), file=sys.stderr)
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
        return 1

    warnings = Counter(str(key[0]) for key in warning_keys)
    total = len(warning_keys)
    for code, count in warnings.most_common():
        print(f"{count:4}  {code}")

    over_budget = {
        code: (count, MAX_WARNINGS_BY_CODE.get(code, 0))
        for code, count in warnings.items()
        if count > MAX_WARNINGS_BY_CODE.get(code, 0)
    }
    if total > MAX_CLIPPY_WARNINGS or over_budget:
        print(
            f"FAIL clippy warnings: {total} > budget {MAX_CLIPPY_WARNINGS}",
            file=sys.stderr,
        )
        for code, (count, budget) in sorted(over_budget.items()):
            print(f"  {code}: {count} > budget {budget}", file=sys.stderr)
        return 1

    print(f"ok   clippy warnings: {total} (budget {MAX_CLIPPY_WARNINGS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
