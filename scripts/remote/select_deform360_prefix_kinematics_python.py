#!/usr/bin/env python3
"""Select a local Python matching the frozen Deform360 GPU runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


_DEFAULT_CANDIDATES = (
    "/home/florianpfaff/.venvs/bpt-gpu/bin/python",
    "/home/github-runner/.venvs/bpt-gpu/bin/python",
    "/home/github-runner/.cache/venvs/bpt-gpu/bin/python",
    "/home/github-runner/miniconda3/envs/bpt-gpu/bin/python",
    "/opt/hostedtoolcache/Python/3.12.3/x64/bin/python",
    "/usr/bin/python3",
)
_EXPECTED_KEYS = (
    "python",
    "numpy",
    "scipy",
    "torch",
    "torch_cuda",
    "warp",
)
_PROBE = r'''
from __future__ import annotations

import importlib.metadata
import json
import sys

import numpy as np
import scipy
import torch
import warp as wp

wp.init()
print(
    json.dumps(
        {
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "warp": wp.__version__,
            "pytest": importlib.metadata.version("pytest"),
            "torch_cuda_available": torch.cuda.is_available(),
            "warp_cuda_device_count": len(wp.get_cuda_devices()),
        },
        sort_keys=True,
    )
)
'''


def _parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict) or any(type(key) is not str for key in payload):
        raise ValueError(f"runtime lock must be a JSON object: {path}")
    return payload


def _expected_runtime(repository_root: Path) -> dict[str, str]:
    path = (
        repository_root
        / "milestones"
        / "deform360-replication-source-backend-v1"
        / "verification"
        / "environment.json"
    )
    payload = _strict_json_object(path)
    expected: dict[str, str] = {}
    for key in _EXPECTED_KEYS:
        value = payload.get(key)
        if type(value) is not str or not value:
            raise ValueError(f"runtime lock {key} must be a nonempty string")
        expected[key] = value
    return expected


def runtime_mismatches(
    expected: Mapping[str, str],
    observed: Mapping[str, Any],
) -> list[str]:
    """Return exact frozen-runtime mismatches for one interpreter probe."""

    mismatches = [
        f"{key}: expected {expected[key]!r}, observed {observed.get(key)!r}"
        for key in _EXPECTED_KEYS
        if observed.get(key) != expected[key]
    ]
    if observed.get("torch_cuda_available") is not True:
        mismatches.append("PyTorch cannot see CUDA")
    device_count = observed.get("warp_cuda_device_count")
    if type(device_count) is not int or device_count < 1:
        mismatches.append("Warp cannot see a CUDA device")
    pytest_version = observed.get("pytest")
    if type(pytest_version) is not str or not pytest_version:
        mismatches.append("pytest is unavailable")
    return mismatches


def _candidate_paths(explicit: Sequence[str]) -> tuple[Path, ...]:
    raw: list[str] = []
    configured = os.environ.get("PREFIX_KINEMATICS_PYTHON")
    if configured:
        raw.append(configured)
    raw.extend(explicit)
    raw.extend(_DEFAULT_CANDIDATES)
    unique: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        # Preserve the invoked path rather than resolving symlinks. A virtual
        # environment's Python executable may point at the base interpreter, but
        # invocation through the venv path is what activates its site-packages.
        path = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
        identity = os.path.normcase(os.fspath(path))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(path)
    return tuple(unique)


def _probe(candidate: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(candidate), "-c", _PROBE],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"probe exited with status {completed.returncode}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("probe produced no JSON output")
    observed = json.loads(lines[-1])
    if not isinstance(observed, dict) or any(type(key) is not str for key in observed):
        raise RuntimeError("probe output is not a JSON object")
    return observed


def _write_report(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    repository_root = args.repository_root.resolve()
    expected = _expected_runtime(repository_root)
    records: list[dict[str, Any]] = []
    selected: Path | None = None
    selected_observed: dict[str, Any] | None = None
    for candidate in _candidate_paths(args.candidate):
        record: dict[str, Any] = {"candidate": str(candidate)}
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            record["status"] = "unavailable"
            records.append(record)
            continue
        try:
            observed = _probe(candidate)
        except (
            OSError,
            RuntimeError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
        ) as error:
            record.update(status="probe_failed", error=str(error))
            records.append(record)
            continue
        mismatches = runtime_mismatches(expected, observed)
        record.update(
            status="matched" if not mismatches else "mismatched",
            observed=observed,
            mismatches=mismatches,
        )
        records.append(record)
        if not mismatches:
            selected = candidate
            selected_observed = observed
            break
    report = {
        "schema_version": 1,
        "artifact_kind": "Deform360PrefixKinematicsPythonSelection",
        "expected": expected,
        "candidates": records,
        "selected": str(selected) if selected is not None else None,
        "selected_runtime": selected_observed,
    }
    _write_report(args.report, report)
    if selected is None:
        rendered = "; ".join(
            f"{record['candidate']}: {record['status']}"
            + (
                f" ({', '.join(record.get('mismatches', []))})"
                if record.get("mismatches")
                else ""
            )
            for record in records
        )
        raise SystemExit(f"no interpreter matches the frozen GPU runtime: {rendered}")
    selected_text = str(selected)
    if re.fullmatch(r"[^\n\r]+", selected_text) is None:
        raise SystemExit("selected interpreter path contains a newline")
    print(selected_text)


if __name__ == "__main__":
    main()
