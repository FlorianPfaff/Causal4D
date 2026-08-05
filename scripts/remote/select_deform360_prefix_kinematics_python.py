#!/usr/bin/env python3
"""Select a local Python matching the Deform360 reproduction runtime."""

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
_REPRODUCTION_RUNTIME_PATH = (
    Path("configs")
    / "causal4d_public"
    / "deform360_source_backend_reproduction_runtime_v1.json"
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
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_file_evidence(repository_root: Path, record: Mapping[str, Any]) -> Path:
    required = frozenset({"path", "sha256", "fact"})
    if "captured_at" in record:
        required |= {"captured_at"}
    _require_exact_fields(
        record,
        required=required,
        name="reproduction-runtime file evidence",
    )
    relative = record.get("path")
    if type(relative) is not str or not relative:
        raise ValueError("reproduction-runtime evidence path must be nonempty")
    fact = record.get("fact")
    if type(fact) is not str or not fact:
        raise ValueError("reproduction-runtime evidence fact must be nonempty")
    path = repository_root / relative
    expected_sha = _require_sha256(
        record.get("sha256"),
        name="reproduction-runtime evidence sha256",
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"reproduction-runtime evidence is not an ordinary file: {path}"
        )
    observed_sha = _sha256_file(path)
    if observed_sha != expected_sha:
        raise ValueError(
            f"reproduction-runtime evidence checksum changed for {path}: "
            f"expected {expected_sha}, observed {observed_sha}"
        )
    return path


def _load_runtime_lock(
    repository_root: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    environment_path = repository_root / _ENVIRONMENT_PATH
    recorded_environment = _strict_json_object(environment_path)
    recorded_runtime: dict[str, str] = {}
    for key in _EXPECTED_KEYS:
        value = recorded_environment.get(key)
        if type(value) is not str or not value:
            raise ValueError(f"recorded runtime {key} must be a nonempty string")
        recorded_runtime[key] = value

    contract_path = repository_root / _REPRODUCTION_RUNTIME_PATH
    contract = _strict_json_object(contract_path)
    _require_exact_fields(
        contract,
        required=frozenset(
            {
                "schema_version",
                "artifact_kind",
                "status",
                "recorded_runtime",
                "candidate_runtime",
                "evidence",
                "boundary",
                "content_sha256",
            }
        ),
        name="reproduction-runtime contract",
    )
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported reproduction-runtime schema version")
    if contract.get("artifact_kind") != (
        "Deform360SourceBackendReproductionRuntimeDeviation"
    ):
        raise ValueError("unsupported reproduction-runtime artifact kind")
    if contract.get("status") != "conditional-reproduction-runtime-deviation":
        raise ValueError("runtime deviation is not conditional reproduction evidence")
    contract_sha256 = _require_sha256(
        contract.get("content_sha256"),
        name="reproduction-runtime content_sha256",
    )
    canonical = dict(contract)
    canonical.pop("content_sha256")
    if _canonical_sha256(canonical) != contract_sha256:
        raise ValueError("reproduction-runtime content checksum changed")

    recorded = contract.get("recorded_runtime")
    if not isinstance(recorded, Mapping):
        raise ValueError("recorded_runtime must be a mapping")
    _require_exact_fields(
        recorded,
        required=frozenset({"path", "sha256", "values"}),
        name="recorded runtime",
    )
    if recorded.get("path") != str(_ENVIRONMENT_PATH):
        raise ValueError("reproduction contract identifies another environment lock")
    environment_sha256 = _require_sha256(
        recorded.get("sha256"),
        name="recorded environment sha256",
    )
    if _sha256_file(environment_path) != environment_sha256:
        raise ValueError("recorded source-backend environment lock changed")
    recorded_values = recorded.get("values")
    if not isinstance(recorded_values, Mapping):
        raise ValueError("recorded runtime values must be a mapping")
    _require_exact_fields(
        recorded_values,
        required=frozenset(_EXPECTED_KEYS),
        name="recorded runtime values",
    )
    for key in _EXPECTED_KEYS:
        if recorded_values.get(key) != recorded_runtime[key]:
            raise ValueError(f"recorded runtime no longer matches {key}")

    candidate = contract.get("candidate_runtime")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate_runtime must be a mapping")
    _require_exact_fields(
        candidate,
        required=frozenset(_EXPECTED_KEYS),
        name="candidate runtime",
    )
    candidate_runtime: dict[str, str] = {}
    for key in _EXPECTED_KEYS:
        value = candidate.get(key)
        if type(value) is not str or not value:
            raise ValueError(f"candidate runtime {key} must be a nonempty string")
        if key != "numpy" and value != recorded_runtime[key]:
            raise ValueError(f"candidate runtime unexpectedly changes {key}")
        candidate_runtime[key] = value
    if candidate_runtime["numpy"] != "1.26.4":
        raise ValueError("candidate NumPy version changed")
    if candidate_runtime["numpy"] == recorded_runtime["numpy"]:
        raise ValueError("candidate runtime does not declare a NumPy deviation")

    evidence = contract.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise ValueError(
            "reproduction-runtime contract must contain exactly three evidence records"
        )
    command_path = _validate_file_evidence(repository_root, evidence[0])
    command_text = command_path.read_text(encoding="utf-8")
    if "/home/florianpfaff/.venvs/bpt-gpu/bin/python -m pytest" not in command_text:
        raise ValueError("runtime evidence lost the archived interpreter command")
    freeze_path = _validate_file_evidence(repository_root, evidence[1])
    freeze_lines = set(freeze_path.read_text(encoding="utf-8").splitlines())
    for line in (
        "numpy==1.26.4",
        "scipy==1.13.1",
        "torch==2.4.0+cu121",
        "warp-lang==1.15.0",
    ):
        if line not in freeze_lines:
            raise ValueError(f"runtime evidence no longer contains {line}")
    workflow_record = evidence[2]
    if not isinstance(workflow_record, Mapping):
        raise ValueError("workflow runtime evidence must be a mapping")
    _require_exact_fields(
        workflow_record,
        required=frozenset(
            {"workflow_run_id", "artifact_id", "artifact_sha256", "fact"}
        ),
        name="workflow runtime evidence",
    )
    if workflow_record.get("workflow_run_id") != 30970401038:
        raise ValueError("workflow runtime identity changed")
    if workflow_record.get("artifact_id") != 8916348471:
        raise ValueError("workflow artifact identity changed")
    _require_sha256(
        workflow_record.get("artifact_sha256"),
        name="workflow artifact sha256",
    )

    boundary = contract.get("boundary")
    expected_boundary = {
        "interpretation_permitted_only_after_zero_baseline_reproduction": True,
        "original_milestone_files_rewritten": False,
        "recorded_runtime_relabelled": False,
        "scientific_artifacts_changed": False,
        "scores_or_decisions_changed": False,
        "target_future_access_permitted": False,
        "target_prefix_access_permitted": False,
        "zero_baseline_reproduction_required": True,
    }
    if boundary != expected_boundary:
        raise ValueError("reproduction-runtime scientific boundary changed")

    provenance = {
        "status": "conditional-reproduction-runtime-deviation",
        "recorded_environment_path": str(_ENVIRONMENT_PATH),
        "recorded_environment_sha256": environment_sha256,
        "recorded_runtime": recorded_runtime,
        "reproduction_runtime_contract_path": str(_REPRODUCTION_RUNTIME_PATH),
        "reproduction_runtime_contract_sha256": contract_sha256,
        "candidate_runtime": candidate_runtime,
        "deviation": {
            "numpy": {
                "recorded": recorded_runtime["numpy"],
                "candidate": candidate_runtime["numpy"],
            }
        },
        "interpretation_permitted_only_after_zero_baseline_reproduction": True,
        "zero_baseline_reproduction_required": True,
    }
    return candidate_runtime, provenance


def _expected_runtime(repository_root: Path) -> dict[str, str]:
    expected, _ = _load_runtime_lock(repository_root)
    return expected


def runtime_mismatches(
    expected: Mapping[str, str],
    observed: Mapping[str, Any],
) -> list[str]:
    """Return exact reproduction-runtime mismatches for one interpreter probe."""

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
        raise SystemExit(f"no interpreter matches the reproduction runtime: {rendered}")
    selected_text = str(selected)
    if re.fullmatch(r"[^\n\r]+", selected_text) is None:
        raise SystemExit("selected interpreter path contains a newline")
    print(selected_text)


if __name__ == "__main__":
    main()
