#!/usr/bin/env python3
"""Enforce kanimiso's direct-dependency and Pure Rust boundary."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MEMBERS = ["signlred", "ojizou-san", "kanimiso"]
DEPENDENCY_KINDS = ("dependencies", "dev-dependencies", "build-dependencies")
FORBID_UNSAFE = re.compile(r"#!\s*\[\s*forbid\s*\(\s*unsafe_code\s*\)\s*\]")
ALLOW_UNSAFE = re.compile(r"#\s*!?\s*\[\s*(?:allow|expect)\s*\(\s*unsafe_code\s*\)")


def load_toml(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    with path.open("rb") as handle:
        return tomllib.load(handle)


def dependency_tables(
    manifest: dict[str, Any], manifest_name: str
) -> list[tuple[str, dict[str, Any]]]:
    """Return every direct dependency table, including target-specific ones."""
    tables: list[tuple[str, dict[str, Any]]] = []
    for kind in DEPENDENCY_KINDS:
        table = manifest.get(kind, {})
        if not isinstance(table, dict):
            raise TypeError(f"{manifest_name} [{kind}] must be a table")
        tables.append((kind, table))

    workspace = manifest.get("workspace", {})
    if isinstance(workspace, dict):
        table = workspace.get("dependencies", {})
        if not isinstance(table, dict):
            raise TypeError(
                f"{manifest_name} [workspace.dependencies] must be a table"
            )
        tables.append(("workspace.dependencies", table))

    targets = manifest.get("target", {})
    if not isinstance(targets, dict):
        raise TypeError(f"{manifest_name} [target] must be a table")
    for target_name, target in targets.items():
        if not isinstance(target, dict):
            raise TypeError(f"{manifest_name} target {target_name!r} must be a table")
        for kind in DEPENDENCY_KINDS:
            table = target.get(kind, {})
            if not isinstance(table, dict):
                raise TypeError(
                    f"{manifest_name} [target.{target_name!r}.{kind}] must be a table"
                )
            tables.append((f"target.{target_name}.{kind}", table))
    return tables


def names(table: Any) -> set[str]:
    return set(table) if isinstance(table, dict) else set()


def format_names(values: set[str]) -> str:
    return "{" + ", ".join(sorted(values)) + "}"


def main() -> int:
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    root = load_toml("Cargo.toml")
    workspace = root.get("workspace", {})
    require(
        workspace.get("members") == EXPECTED_MEMBERS,
        "workspace.members must be exactly " + repr(EXPECTED_MEMBERS),
    )
    require(
        workspace.get("default-members") == EXPECTED_MEMBERS,
        "workspace.default-members must be exactly " + repr(EXPECTED_MEMBERS),
    )

    workspace_dependencies = workspace.get("dependencies", {})
    expected_workspace_dependencies = {"faer", "ndarray", "signlred", "ojizou-san"}
    require(
        names(workspace_dependencies) == expected_workspace_dependencies,
        "workspace dependencies must be exactly "
        + format_names(expected_workspace_dependencies),
    )

    expected_faer = {
        "version": "=0.24.4",
        "default-features": False,
        "features": ["std", "linalg", "rayon"],
    }
    require(
        workspace_dependencies.get("faer") == expected_faer,
        "workspace faer must be exactly version 0.24.4 with only "
        "std,linalg,rayon features and default features disabled",
    )
    expected_ndarray = {
        "version": "=0.17.2",
        "default-features": False,
        "features": ["std"],
    }
    require(
        workspace_dependencies.get("ndarray") == expected_ndarray,
        "workspace ndarray must be exactly version 0.17.2 with only the std "
        "feature and default features disabled",
    )
    require(
        workspace_dependencies.get("signlred")
        == {"path": "signlred", "version": "0.1.0"},
        "workspace signlred must remain the local signlred path dependency",
    )
    require(
        workspace_dependencies.get("ojizou-san")
        == {"path": "ojizou-san", "version": "0.1.0"},
        "workspace ojizou-san must remain the local ojizou-san path dependency",
    )

    policies = {
        "kanimiso/Cargo.toml": {
            "package": "kanimiso",
            "dependencies": {"faer", "ndarray", "signlred", "ojizou-san"},
            "dev-dependencies": {"serde_json"},
            "build-dependencies": set(),
        },
        "signlred/Cargo.toml": {
            "package": "signlred",
            "dependencies": set(),
            "dev-dependencies": set(),
            "build-dependencies": set(),
        },
        "ojizou-san/Cargo.toml": {
            "package": "ojizou-san",
            "dependencies": {"signlred"},
            "dev-dependencies": set(),
            "build-dependencies": set(),
        },
    }
    manifests = {"Cargo.toml": root}
    for manifest_path, policy in policies.items():
        manifest = load_toml(manifest_path)
        manifests[manifest_path] = manifest
        require(
            manifest.get("package", {}).get("name") == policy["package"],
            f"{manifest_path} package name must be {policy['package']!r}",
        )
        for kind in DEPENDENCY_KINDS:
            expected = policy[kind]
            actual = names(manifest.get(kind, {}))
            require(
                actual == expected,
                f"{manifest_path} [{kind}] must be exactly {format_names(expected)}; "
                f"found {format_names(actual)}",
            )

        for target_name, target in manifest.get("target", {}).items():
            for kind in DEPENDENCY_KINDS:
                target_dependencies = names(target.get(kind, {}))
                require(
                    not target_dependencies,
                    f"{manifest_path} target {target_name!r} [{kind}] must be empty; "
                    f"found {format_names(target_dependencies)}",
                )

    workspace_reference = {"workspace": True}
    for dependency in ("faer", "ndarray", "signlred", "ojizou-san"):
        require(
            manifests["kanimiso/Cargo.toml"]
            .get("dependencies", {})
            .get(dependency)
            == workspace_reference,
            f"kanimiso dependency {dependency!r} must use only workspace = true",
        )
    require(
        manifests["ojizou-san/Cargo.toml"]
        .get("dependencies", {})
        .get("signlred")
        == workspace_reference,
        "ojizou-san dependency 'signlred' must use only workspace = true",
    )

    serde_json = manifests["kanimiso/Cargo.toml"].get("dev-dependencies", {}).get(
        "serde_json"
    )
    require(
        serde_json == "1.0",
        "kanimiso serde_json must remain a crates.io-only dev dependency at version 1.0",
    )

    for manifest_name, manifest in manifests.items():
        require(
            "patch" not in manifest and "replace" not in manifest,
            f"{manifest_name} must not override registry dependencies with "
            "[patch] or [replace]",
        )
        try:
            tables = dependency_tables(manifest, manifest_name)
        except TypeError as error:
            failures.append(str(error))
            continue
        for table_name, table in tables:
            for dependency, specification in table.items():
                if isinstance(specification, dict):
                    require(
                        "git" not in specification,
                        f"{manifest_name} [{table_name}] dependency "
                        f"{dependency!r} must not use git",
                    )

    lock = load_toml("Cargo.lock")
    locked_faer = [
        package.get("version")
        for package in lock.get("package", [])
        if package.get("name") == "faer"
    ]
    require(
        locked_faer == ["0.24.4"],
        f"Cargo.lock must contain exactly faer 0.24.4; found {locked_faer!r}",
    )
    locked_ndarray = [
        package.get("version")
        for package in lock.get("package", [])
        if package.get("name") == "ndarray"
    ]
    require(
        locked_ndarray == ["0.17.2"],
        "Cargo.lock must contain exactly ndarray 0.17.2; "
        f"found {locked_ndarray!r}",
    )
    forbidden_native_packages = {
        "accelerate-src",
        "blas-src",
        "blas-sys",
        "cblas-sys",
        "intel-mkl-src",
        "lapack-sys",
        "netlib-src",
        "openblas-src",
    }
    locked_names = {
        package.get("name")
        for package in lock.get("package", [])
        if isinstance(package.get("name"), str)
    }
    found_native_packages = forbidden_native_packages & locked_names
    require(
        not found_native_packages,
        "Cargo.lock must not contain native BLAS/LAPACK backends; found "
        + format_names(found_native_packages),
    )

    crate_roots = [
        "kanimiso/src/lib.rs",
        "signlred/src/lib.rs",
        "ojizou-san/src/lib.rs",
    ]
    for relative_path in crate_roots:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        require(
            len(FORBID_UNSAFE.findall(text)) == 1,
            f"{relative_path} must contain exactly one #![forbid(unsafe_code)]",
        )

    for source_root in ("kanimiso/src", "signlred/src", "ojizou-san/src"):
        for path in (ROOT / source_root).rglob("*.rs"):
            text = path.read_text(encoding="utf-8")
            require(
                ALLOW_UNSAFE.search(text) is None,
                f"{path.relative_to(ROOT)} must not allow or expect unsafe_code",
            )

    if failures:
        for failure in failures:
            print(f"FAIL dependency policy: {failure}", file=sys.stderr)
        return 1

    print(
        "ok   dependency policy: ndarray 0.17.2 and faer 0.24.4 are the only "
        "external runtime scientific dependencies; native BLAS, workspace, "
        "dependency, and unsafe guards match"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
