#!/usr/bin/env python3
"""Run the locked source-only Deform360 filament-support diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from collections.abc import Callable, Mapping
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np

from causal4d.atomic_io import atomic_write_json
from causal4d_public.deform360_filament_support import (
    run_source_filament_support_diagnostic,
    validate_source_filament_support_diagnostic,
)


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--lock",
        type=Path,
        default=(
            repository_root
            / "configs"
            / "causal4d_public"
            / "deform360_filament_support_v1.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            repository_root
            / "configs"
            / "causal4d_public"
            / "deform360_replication_v1.json"
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_prefix_runtime_module(repository_root: Path) -> ModuleType:
    path = repository_root / "scripts/remote/run_deform360_prefix_kinematics.py"
    spec = importlib.util.spec_from_file_location(
        "causal4d_filament_support_runtime_contract",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load the prefix runtime contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_provenance(
    repository_root: Path,
    result_path: Path,
    result_sha256: str,
    *,
    runtime_selection_path: Path,
    runtime_selection: Mapping[str, Any],
) -> dict[str, Any]:
    environment_path = (
        repository_root
        / "milestones"
        / "deform360-replication-source-backend-v1"
        / "verification"
        / "environment.json"
    )
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "artifact_kind": "Deform360FilamentSupportRuntime",
        "result_path": str(result_path),
        "result_file_sha256": _sha256_file(result_path),
        "result_sha256": result_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "runtime_selection_path": str(runtime_selection_path),
        "runtime_selection_file_sha256": _sha256_file(runtime_selection_path),
        "runtime_lock_provenance": runtime_selection["runtime_provenance"],
        "dataset_revision": environment["dataset_revision"],
        "repository_commit": _command_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"]
        ),
        "distributions": {
            name: _distribution_version(name) for name in ("causal4d", "numpy", "scipy")
        },
        "information_boundary": {
            "source_structure_only": True,
            "future_mechanics_scores_read": False,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
        },
    }


def main() -> None:
    args = _parse_args()
    repository_root = args.repository_root.resolve()
    output = args.output.resolve()
    runtime_selection_path = args.runtime_selection.resolve()
    prefix_runtime = _load_prefix_runtime_module(repository_root)
    load_runtime_selection = cast(
        Callable[[Path], dict[str, Any]],
        getattr(prefix_runtime, "_load_runtime_selection"),
    )
    runtime_selection = load_runtime_selection(runtime_selection_path)
    result = run_source_filament_support_diagnostic(
        repository_root,
        args.protocol,
        args.data_root,
        output,
        lock_path=args.lock,
    )
    validate_source_filament_support_diagnostic(result)
    runtime_path = output.with_name(f"{output.stem}.runtime.json")
    atomic_write_json(
        runtime_path,
        _runtime_provenance(
            repository_root,
            output,
            result["result_sha256"],
            runtime_selection_path=runtime_selection_path,
            runtime_selection=runtime_selection,
        ),
    )
    print(
        json.dumps(
            {
                "classification": result["decision"]["classification"],
                "decision_passed": result["decision"]["passed"],
                "reset_count": len(result["reset_records"]),
                "result_path": str(output),
                "result_sha256": result["result_sha256"],
                "runtime_path": str(runtime_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
