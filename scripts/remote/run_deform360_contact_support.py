#!/usr/bin/env python3
"""Run the locked source-only Deform360 contact/support diagnostic."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from causal4d.atomic_io import atomic_write_json
from causal4d_public.deform360_contact_support_diagnostic import (
    run_source_contact_support_diagnostic,
    validate_source_contact_support_diagnostic,
)
from run_deform360_prefix_kinematics import (
    _load_runtime_selection,
    _sha256_file,
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
            / "deform360_contact_support_v1.json"
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
        "artifact_kind": "Deform360ContactSupportRuntime",
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
    result = run_source_contact_support_diagnostic(
        repository_root,
        args.protocol,
        args.data_root,
        args.official_phystwin_repo,
        output,
        lock_path=args.lock,
        device=args.device,
    )
    validate_source_contact_support_diagnostic(result)
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
                "any_candidate_supported": result["decision"][
                    "any_candidate_supported"
                ],
                "episode_count": len(result["episode_records"]),
                "result_path": str(output),
                "result_sha256": result["result_sha256"],
                "runtime_path": str(runtime_path),
                "runtime_selection_path": str(runtime_selection_path),
                "supported_candidate_policies": result["decision"][
                    "supported_candidate_policies"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
