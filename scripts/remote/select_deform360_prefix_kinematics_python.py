#!/usr/bin/env python3
"""Select a local Python matching the frozen Deform360 GPU runtime."""

from __future__ import annotations

import argparse
import hashlib
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
_ENVIRONMENT_PATH = (
    Path("milestones")
    / "deform360-replication-source-backend-v1"
    / "verification"
    / "environment.json"
)
_ERRATUM_PATH = (
    Path("configs")
    / "causal4d_public"
    / "deform360_source_backend_runtime_erratum_v1.json"
)
_PROBE = r"""
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
"""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_exact_fields(
    payload: Mapping[str, Any],
    *,
    required: frozenset[str],
    name: str,
) -> None:
    actual = set(payload)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_file_evidence(repository_root: Path, record: Mapping[str, Any]) -> Path:
    _require_exact_fields(
        record,
        required=frozenset({"path", "sha256", "fact"})
        | ({"captured_at"} if "captured_at" in record else frozenset()),
        name="runtime-erratum file evidence",
    )
    relative = record.get("path")
    if type(relative) is not str or not relative:
        raise ValueError("runtime-erratum evidence path must be a nonempty string")
    path = repository_root / relative
    expected_sha = _require_sha256(
        record.get("sha256"),
        name="runtime-erratum evidence sha256",
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"runtime-erratum evidence is not an ordinary file: {path}")
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha:
        raise ValueError(
            f"runtime-erratum evidence checksum changed for {path}: "
            f"expected {expected_sha}, observed {observed_sha}"
        )
    return path


def _load_runtime_lock(
    repository_root: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    environment_path = repository_root / _ENVIRONMENT_PATH
    payload = _strict_json_object(environment_path)
    expected: dict[str, str] = {}
    for key in _EXPECTED_KEYS:
        value = payload.get(key)
        if type(value) is not str or not value:
            raise ValueError(f"runtime lock {key} must be a nonempty string")
        expected[key] = value

    erratum_path = repository_root / _ERRATUM_PATH
    erratum = _strict_json_object(erratum_path)
    _require_exact_fields(
        erratum,
        required=frozenset(
            {
                "schema_version",
                "artifact_kind",
                "status",
                "original_environment",
                "corrections",
                "unchanged_runtime",
                "evidence",
                "boundary",
                "content_sha256",
            }
        ),
        name="runtime erratum",
    )
    if erratum.get("schema_version") != 1:
        raise ValueError("unsupported runtime-erratum schema version")
    if erratum.get("artifact_kind") != "Deform360SourceBackendRuntimeErratum":
        raise ValueError("unsupported runtime-erratum artifact kind")
    if erratum.get("status") != "additive-provenance-correction":
        raise ValueError("runtime erratum is not an additive provenance correction")
    recorded_content_sha = _require_sha256(
        erratum.get("content_sha256"),
        name="runtime-erratum content_sha256",
    )
    canonical = dict(erratum)
    canonical.pop("content_sha256")
    if _canonical_sha256(canonical) != recorded_content_sha:
        raise ValueError("runtime-erratum content checksum changed")

    original = erratum.get("original_environment")
    if not isinstance(original, Mapping):
        raise ValueError("runtime-erratum original_environment must be a mapping")
    _require_exact_fields(
        original,
        required=frozenset({"path", "sha256", "recorded_numpy"}),
        name="runtime-erratum original_environment",
    )
    if original.get("path") != str(_ENVIRONMENT_PATH):
        raise ValueError("runtime erratum identifies another environment lock")
    expected_environment_sha = _require_sha256(
        original.get("sha256"),
        name="original environment sha256",
    )
    if _sha256_file(environment_path) != expected_environment_sha:
        raise ValueError("original source-backend environment lock changed")
    if original.get("recorded_numpy") != expected["numpy"]:
        raise ValueError("runtime erratum no longer matches the recorded NumPy value")

    corrections = erratum.get("corrections")
    if not isinstance(corrections, Mapping) or set(corrections) != {"numpy"}:
        raise ValueError("runtime erratum may correct only the NumPy field")
    numpy_correction = corrections["numpy"]
    if not isinstance(numpy_correction, Mapping):
        raise ValueError("runtime-erratum NumPy correction must be a mapping")
    _require_exact_fields(
        numpy_correction,
        required=frozenset({"recorded", "corrected"}),
        name="runtime-erratum NumPy correction",
    )
    if numpy_correction.get("recorded") != expected["numpy"]:
        raise ValueError("runtime-erratum recorded NumPy value changed")
    corrected_numpy = numpy_correction.get("corrected")
    if corrected_numpy != "1.26.4":
        raise ValueError("runtime-erratum corrected NumPy value changed")

    unchanged = erratum.get("unchanged_runtime")
    if not isinstance(unchanged, Mapping):
        raise ValueError("runtime-erratum unchanged_runtime must be a mapping")
    _require_exact_fields(
        unchanged,
        required=frozenset(set(_EXPECTED_KEYS) - {"numpy"}),
        name="runtime-erratum unchanged runtime",
    )
    for key, value in unchanged.items():
        if value != expected[key]:
            raise ValueError(f"runtime erratum unexpectedly changes {key}")

    evidence = erratum.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise ValueError("runtime erratum must contain exactly three evidence records")
    command_path = _validate_file_evidence(repository_root, evidence[0])
    command_text = command_path.read_text(encoding="utf-8")
    if "/home/florianpfaff/.venvs/bpt-gpu/bin/python -m pytest" not in command_text:
        raise ValueError("runtime erratum lost the archived interpreter command")
    freeze_path = _validate_file_evidence(repository_root, evidence[1])
    freeze_lines = set(freeze_path.read_text(encoding="utf-8").splitlines())
    for line in (
        "numpy==1.26.4",
        "scipy==1.13.1",
        "torch==2.4.0+cu121",
        "warp-lang==1.15.0",
    ):
        if line not in freeze_lines:
            raise ValueError(f"runtime erratum evidence no longer contains {line}")
    workflow_record = evidence[2]
    if not isinstance(workflow_record, Mapping):
        raise ValueError("runtime-erratum workflow evidence must be a mapping")
    _require_exact_fields(
        workflow_record,
        required=frozenset(
            {"workflow_run_id", "artifact_id", "artifact_sha256", "fact"}
        ),
        name="runtime-erratum workflow evidence",
    )
    if workflow_record.get("workflow_run_id") != 30970401038:
        raise ValueError("runtime-erratum workflow identity changed")
    if workflow_record.get("artifact_id") != 8916348471:
        raise ValueError("runtime-erratum artifact identity changed")
    _require_sha256(
        workflow_record.get("artifact_sha256"),
        name="runtime-erratum workflow artifact sha256",
    )

    boundary = erratum.get("boundary")
    expected_boundary = {
        "original_milestone_files_rewritten": False,
        "scientific_artifacts_changed": False,
        "scores_or_decisions_changed": False,
        "target_future_access_permitted": False,
        "target_prefix_access_permitted": False,
        "zero_baseline_reproduction_required": True,
    }
    if boundary != expected_boundary:
        raise ValueError("runtime-erratum scientific boundary changed")

    expected["numpy"] = corrected_numpy
    provenance = {
        "environment_path": str(_ENVIRONMENT_PATH),
        "environment_sha256": expected_environment_sha,
        "erratum_path": str(_ERRATUM_PATH),
        "erratum_sha256": recorded_content_sha,
        "correction": {"numpy": {"recorded": "2.5.1", "effective": "1.26.4"}},
        "zero_baseline_reproduction_required": True,
    }
    return expected, provenance


def _expected_runtime(repository_root: Path) -> dict[str, str]:
    expected, _ = _load_runtime_lock(repository_root)
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
    expected, runtime_provenance = _load_runtime_lock(repository_root)
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
        "runtime_provenance": runtime_provenance,
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
