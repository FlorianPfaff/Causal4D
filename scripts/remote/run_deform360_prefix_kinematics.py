#!/usr/bin/env python3
"""Run the locked source-only Deform360 prefix-kinematics diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from causal4d.atomic_io import atomic_write_json
from causal4d_public.deform360_prefix_kinematics_diagnostic import (
    run_source_prefix_kinematics_diagnostic,
    validate_source_prefix_kinematics_diagnostic,
)


_RUNTIME_KEYS = ("python", "numpy", "scipy", "torch", "torch_cuda", "warp")


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
            / "deform360_prefix_kinematics_v1.json"
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
    parser.add_argument("--bayesian-phystwin-repo", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate runtime-selection JSON key {key!r}")
        result[key] = value
    return result


def _load_runtime_selection(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("runtime selection must be an ordinary file")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite runtime-selection value {value!r}")
        ),
    )
    if not isinstance(payload, dict) or any(type(key) is not str for key in payload):
        raise ValueError("runtime selection must be a string-keyed JSON object")
    required = {
        "schema_version",
        "artifact_kind",
        "expected",
        "runtime_provenance",
        "candidates",
        "selected",
        "selected_runtime",
    }
    if set(payload) != required:
        raise ValueError("runtime-selection fields changed")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported runtime-selection schema version")
    if payload["artifact_kind"] != "Deform360PrefixKinematicsPythonSelection":
        raise ValueError("unsupported runtime-selection artifact kind")
    selected = payload["selected"]
    if (
        type(selected) is not str
        or Path(selected).absolute() != Path(sys.executable).absolute()
    ):
        raise ValueError("runtime selection identifies another interpreter")
    expected = payload["expected"]
    observed = payload["selected_runtime"]
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise ValueError("runtime selection omitted expected or observed runtime")
    for key in _RUNTIME_KEYS:
        if expected.get(key) != observed.get(key):
            raise ValueError(f"runtime selection does not match {key}")
    if observed.get("torch_cuda_available") is not True:
        raise ValueError("selected PyTorch runtime cannot see CUDA")
    warp_device_count = observed.get("warp_cuda_device_count")
    if type(warp_device_count) is not int or warp_device_count < 1:
        raise ValueError("selected Warp runtime cannot see CUDA")

    provenance = payload["runtime_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("runtime selection omitted lock provenance")
    if provenance.get("status") != "conditional-reproduction-runtime-deviation":
        raise ValueError("runtime selection has another provenance status")
    if provenance.get("zero_baseline_reproduction_required") is not True:
        raise ValueError("runtime selection relaxed zero-baseline reproduction")
    if (
        provenance.get(
            "interpretation_permitted_only_after_zero_baseline_reproduction"
        )
        is not True
    ):
        raise ValueError("runtime selection permits premature interpretation")
    recorded = provenance.get("recorded_runtime")
    candidate = provenance.get("candidate_runtime")
    if not isinstance(recorded, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("runtime selection omitted recorded or candidate runtime")
    if dict(candidate) != dict(expected):
        raise ValueError("runtime selection candidate runtime changed")
    if recorded.get("numpy") != "2.5.1" or candidate.get("numpy") != "1.26.4":
        raise ValueError("runtime selection has another NumPy deviation")
    for key in _RUNTIME_KEYS:
        if key != "numpy" and recorded.get(key) != candidate.get(key):
            raise ValueError(f"runtime selection unexpectedly changes {key}")
    if provenance.get("deviation") != {
        "numpy": {"candidate": "1.26.4", "recorded": "2.5.1"}
    }:
        raise ValueError("runtime selection has another declared deviation")
    return payload


def _runtime_provenance(
    repository_root: Path,
    result_path: Path,
    result_sha256: str,
    *,
    bayesian_phystwin_repo: Path,
    deform360_repo: Path,
    official_phystwin_repo: Path,
    runtime_selection_path: Path,
    runtime_selection: Mapping[str, Any],
) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360PrefixKinematicsRuntime",
        "result_path": str(result_path),
        "result_file_sha256": _sha256_file(result_path),
        "result_sha256": result_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "runtime_selection_path": str(runtime_selection_path),
        "runtime_selection_file_sha256": _sha256_file(runtime_selection_path),
        "runtime_lock_provenance": runtime_selection["runtime_provenance"],
        "repositories": {
            name: {
                "path": str(path.resolve()),
                "commit": _command_output(
                    ["git", "-C", str(path.resolve()), "rev-parse", "HEAD"]
                ),
            }
            for name, path in (
                ("causal4d", repository_root),
                ("bayesian_phystwin", bayesian_phystwin_repo),
                ("deform360", deform360_repo),
                ("official_phystwin", official_phystwin_repo),
            )
        },
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "distributions": {
            name: _distribution_version(name)
            for name in (
                "bayesian-phystwin",
                "numpy",
                "scipy",
                "torch",
                "warp-lang",
            )
        },
    }
    try:
        import torch
    except ImportError:
        runtime["torch_runtime"] = None
    else:
        runtime["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
    try:
        import warp as wp
    except ImportError:
        runtime["warp_runtime"] = None
    else:
        runtime["warp_runtime"] = {
            "version": getattr(wp, "__version__", None),
            "devices": [str(device) for device in wp.get_devices()],
        }
    return runtime


def main() -> None:
    args = _parse_args()
    repository_root = args.repository_root.resolve()
    output = args.output.resolve()
    runtime_selection_path = args.runtime_selection.resolve()
    runtime_selection = _load_runtime_selection(runtime_selection_path)
    result = run_source_prefix_kinematics_diagnostic(
        repository_root,
        args.protocol,
        args.data_root,
        args.official_phystwin_repo,
        output,
        lock_path=args.lock,
        device=args.device,
    )
    validate_source_prefix_kinematics_diagnostic(result)
    runtime_path = output.with_name(f"{output.stem}.runtime.json")
    atomic_write_json(
        runtime_path,
        _runtime_provenance(
            repository_root,
            output,
            result["result_sha256"],
            bayesian_phystwin_repo=args.bayesian_phystwin_repo.resolve(),
            deform360_repo=args.deform360_repo.resolve(),
            official_phystwin_repo=args.official_phystwin_repo.resolve(),
            runtime_selection_path=runtime_selection_path,
            runtime_selection=runtime_selection,
        ),
    )
    print(
        json.dumps(
            {
                "decision_passed": result["decision"]["passed"],
                "episode_count": len(result["episode_records"]),
                "result_path": str(output),
                "result_sha256": result["result_sha256"],
                "runtime_path": str(runtime_path),
                "runtime_selection_path": str(runtime_selection_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
