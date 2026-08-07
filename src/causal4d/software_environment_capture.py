"""Capture exact software evidence for the confirmatory acquisition gate."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name

from causal4d.atomic_io import atomic_write_json
from causal4d.preacquisition_readiness_contracts import (
    GATE_EVIDENCE_ARTIFACT_KIND,
    GATE_EVIDENCE_SCHEMA_VERSION,
    GATE_PATHS,
    _read_json_mapping,
    load_registered_preacquisition_chain,
)
from causal4d.real_evidence_contract_v2 import build_real_evidence_status
from causal4d.stack_lock import (
    STACK_PIPELINE,
    inspect_wheel,
    load_stack_lock,
    verify_stack_lock,
)

CAPTURE_SCHEMA_NAME = "causal4d.software-environment-capture"
CAPTURE_SCHEMA_VERSION = 1
CAPTURE_GENERATOR = "causal4d protocol readiness capture-software-environment"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTAINER_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class _CaptureContext:
    protocol: Mapping[str, Any]
    v4: Mapping[str, Any]
    method_freeze: Mapping[str, Any]
    method_freeze_sha256: str
    method_freeze_validation_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _safe_nonempty(value: Any, *, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} is required")
    return value.strip()


def _utc_timestamp(value: str | None) -> str:
    text = value or datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("completed_at_utc must be ISO 8601") from error
    _require(parsed.tzinfo is not None, "completed_at_utc must include a timezone")
    _require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        "completed_at_utc must be UTC",
    )
    return text


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    candidate = Path(path)
    _require(candidate.is_file(), f"{name} is missing: {candidate}")
    _require(not candidate.is_symlink(), f"{name} must not be a symlink")
    return candidate.resolve(strict=True)


def _descriptor(path: Path, *, relative_path: str) -> dict[str, Any]:
    digest, size = _sha256_file(path)
    return {"path": relative_path, "sha256": digest, "bytes": size}


def _archive_sha256(direct_url: Mapping[str, Any]) -> str | None:
    archive = direct_url.get("archive_info")
    if not isinstance(archive, Mapping):
        return None
    raw_hash = archive.get("hash")
    if isinstance(raw_hash, str) and raw_hash.startswith("sha256="):
        digest = raw_hash.removeprefix("sha256=")
        if _SHA256.fullmatch(digest):
            return digest
    hashes = archive.get("hashes")
    if isinstance(hashes, Mapping):
        digest = hashes.get("sha256")
        if isinstance(digest, str) and _SHA256.fullmatch(digest):
            return digest
    return None


def _distribution_record(distribution: metadata.Distribution) -> dict[str, Any]:
    raw_name = distribution.metadata.get("Name")
    _require(isinstance(raw_name, str) and bool(raw_name), "distribution has no Name")
    name = canonicalize_name(raw_name)
    direct_url: dict[str, Any] | None = None
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is not None:
        try:
            parsed = json.loads(direct_url_text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} has invalid direct_url.json") from error
        _require(isinstance(parsed, Mapping), f"{name} direct_url.json is invalid")
        direct_url = dict(parsed)
    editable = bool(
        isinstance(direct_url, Mapping)
        and isinstance(direct_url.get("dir_info"), Mapping)
        and direct_url["dir_info"].get("editable") is True
    )
    return {
        "name": name,
        "version": distribution.version,
        "installer": (distribution.read_text("INSTALLER") or "").strip() or None,
        "editable": editable,
        "direct_url": direct_url,
        "archive_sha256": (
            _archive_sha256(direct_url) if direct_url is not None else None
        ),
    }


def _installed_distribution_records() -> tuple[dict[str, Any], ...]:
    records: dict[str, dict[str, Any]] = {}
    for distribution in metadata.distributions():
        record = _distribution_record(distribution)
        name = str(record["name"])
        _require(name not in records, f"duplicate installed distribution: {name}")
        records[name] = record
    return tuple(records[name] for name in sorted(records))


def _validate_installed_core_stack(
    lock: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    installed = {str(record["name"]): record for record in records}
    validated: list[dict[str, Any]] = []
    for entry in lock["distributions"]:
        name = str(entry["name"])
        record = installed.get(name)
        _require(record is not None, f"locked distribution is not installed: {name}")
        _require(record.get("editable") is False, f"{name} is installed editable")
        _require(
            record.get("version") == entry["version"],
            f"installed {name} version differs from the locked wheel",
        )
        expected_digest = entry["wheel"]["sha256"]
        _require(
            record.get("archive_sha256") == expected_digest,
            f"installed {name} is not bound to the exact locked wheel archive",
        )
        validated.append(
            {
                "name": name,
                "version": record["version"],
                "wheel_sha256": expected_digest,
                "installed_archive_sha256": record["archive_sha256"],
                "editable": False,
            }
        )
    return tuple(validated)


def _optional_version(*distribution_names: str) -> str | None:
    for name in distribution_names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _cuda_runtime_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    value = getattr(getattr(torch, "version", None), "cuda", None)
    return str(value) if value else None


def _cuda_driver_version() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("cuda backend requires a working nvidia-smi driver query") from error
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    _require(bool(versions), "nvidia-smi returned no CUDA driver version")
    _require(len(versions) == 1, "CUDA devices report different driver versions")
    return versions.pop()


def _runtime_snapshot(
    *,
    execution_backend: str,
    container_image_digest: str | None,
    installed_records: Sequence[Mapping[str, Any]],
    core_stack: Sequence[Mapping[str, Any]],
    captured_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(
        execution_backend in {"numpy_cpu", "warp_cpu", "cuda"},
        "execution_backend must be numpy_cpu, warp_cpu, or cuda",
    )
    if container_image_digest is not None:
        _require(
            _CONTAINER_DIGEST.fullmatch(container_image_digest) is not None,
            "container_image_digest must be sha256:<64 lowercase hex>",
        )
    try:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("resolved environment cannot run pip check") from error
    _require(
        pip_check.returncode == 0,
        "resolved environment failed pip check: "
        + (pip_check.stdout + pip_check.stderr).strip(),
    )
    editable = sorted(
        str(record["name"]) for record in installed_records if record.get("editable")
    )
    _require(not editable, f"editable distributions are forbidden: {editable}")

    python_record = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(aliased=True, terse=False),
    }
    numpy_version = _optional_version("numpy")
    scipy_version = _optional_version("scipy")
    _require(numpy_version is not None, "numpy is not installed")
    _require(scipy_version is not None, "scipy is not installed")
    torch_version = _optional_version("torch")
    warp_version = _optional_version("warp-lang")
    opencv_version = _optional_version(
        "opencv-python-headless",
        "opencv-python",
        "opencv-contrib-python-headless",
        "opencv-contrib-python",
    )
    if execution_backend in {"warp_cpu", "cuda"}:
        _require(torch_version is not None, f"{execution_backend} requires torch")
        _require(warp_version is not None, f"{execution_backend} requires warp-lang")
    cuda_runtime = None
    cuda_driver = None
    if execution_backend == "cuda":
        cuda_runtime = _cuda_runtime_version()
        _require(cuda_runtime is not None, "cuda backend requires torch CUDA runtime")
        cuda_driver = _cuda_driver_version()

    runtime_record = {
        "resolved_dependency_report": None,
        "execution_backend": execution_backend,
        "containerized": container_image_digest is not None,
        "container_image_digest": container_image_digest,
        "numpy_version": numpy_version,
        "scipy_version": scipy_version,
        "torch_version": torch_version,
        "warp_version": warp_version,
        "opencv_version": opencv_version,
        "cuda_runtime_version": cuda_runtime,
        "cuda_driver_version": cuda_driver,
    }
    report = {
        "schema_name": CAPTURE_SCHEMA_NAME,
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "generated_by": CAPTURE_GENERATOR,
        "captured_at_utc": captured_at_utc,
        "python_executable": str(Path(sys.executable).resolve()),
        "python": python_record,
        "runtime_environment": deepcopy(runtime_record),
        "pip_check": {
            "returncode": pip_check.returncode,
            "stdout": pip_check.stdout.strip(),
            "stderr": pip_check.stderr.strip(),
        },
        "core_stack": [dict(record) for record in core_stack],
        "distributions": [dict(record) for record in installed_records],
    }
    return python_record, runtime_record, report


def _capture_context(
    repository_root: Path,
    dataset_root: Path,
) -> _CaptureContext:
    protocol, _, _, v4 = load_registered_preacquisition_chain(repository_root)
    real_status = build_real_evidence_status(
        protocol,
        dataset_root,
        repository_root=repository_root,
        verify_file_hashes=True,
    )
    prerequisites = real_status.get("prerequisites")
    _require(isinstance(prerequisites, Mapping), "real evidence prerequisites are missing")
    freeze = prerequisites.get("method_freeze")
    attestation = prerequisites.get("method_freeze_validation")
    _require(isinstance(freeze, Mapping) and freeze.get("valid") is True, "method freeze is not valid")
    _require(
        isinstance(attestation, Mapping) and attestation.get("valid") is True,
        "method freeze attestation is not valid",
    )
    for field in ("manifest_executions", "acquired_executions", "validated_executions"):
        _require(real_status.get(field, 0) == 0, "confirmatory collection has already started")
    freeze_path = dataset_root / "method_freeze.json"
    method_freeze = _read_json_mapping(freeze_path, name="method freeze")
    return _CaptureContext(
        protocol=protocol,
        v4=v4,
        method_freeze=method_freeze,
        method_freeze_sha256=str(freeze["sha256"]),
        method_freeze_validation_sha256=str(attestation["sha256"]),
    )


def _validate_gate_template(
    gate: Mapping[str, Any],
    context: _CaptureContext,
) -> None:
    _require(
        gate.get("schema_version") == GATE_EVIDENCE_SCHEMA_VERSION,
        "unsupported software-environment gate schema",
    )
    _require(
        gate.get("artifact_kind") == GATE_EVIDENCE_ARTIFACT_KIND,
        "unexpected software-environment gate artifact kind",
    )
    _require(gate.get("gate_id") == "software_environment_locked", "wrong gate template")
    _require(gate.get("status") == "template", "software-environment gate is already sealed")
    _require(gate.get("artifact_sha256") is None, "gate template already has a digest")
    _require(gate.get("protocol_id") == context.protocol["protocol_id"], "gate protocol mismatch")
    _require(
        gate.get("protocol_design_sha256") == context.protocol["design_sha256"],
        "gate protocol digest mismatch",
    )
    _require(gate.get("preacquisition_plan_id") == context.v4["plan_id"], "gate plan mismatch")
    _require(
        gate.get("preacquisition_amendment_sha256") == context.v4["amendment_sha256"],
        "gate amendment mismatch",
    )
    approval = gate.get("approval")
    _require(
        isinstance(approval, Mapping) and approval.get("approved") is False,
        "gate template approval is invalid",
    )


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as output:
        shutil.copyfileobj(source_handle, output, length=1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    source_digest, source_size = _sha256_file(source)
    copied_digest, copied_size = _sha256_file(destination)
    _require(
        (source_digest, source_size) == (copied_digest, copied_size),
        f"copied artifact changed bytes: {source.name}",
    )


def _portable_verification_report(
    report: Mapping[str, Any],
    relative_wheel_paths: Mapping[str, str],
) -> dict[str, Any]:
    portable = deepcopy(dict(report))
    wheel_set = portable.get("wheel_set")
    if isinstance(wheel_set, Mapping):
        entries = wheel_set.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and entry.get("name") in relative_wheel_paths:
                    entry["path"] = relative_wheel_paths[str(entry["name"])]
    return portable


def capture_software_environment_template(
    repository_root: str | Path,
    dataset_root: str | Path,
    stack_lock_path: str | Path,
    wheel_paths: Sequence[str | Path],
    *,
    execution_backend: str,
    observation_producer_name: str,
    observation_producer_version: str,
    observation_artifact_contract: str,
    prob4d_used: bool,
    prob4d_unused_reason: str | None = None,
    prob4d_observation_contract_version: str | None = None,
    container_image_digest: str | None = None,
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Populate, but do not approve, the software-environment gate template."""

    repository = Path(repository_root).resolve(strict=True)
    dataset = Path(dataset_root).resolve(strict=True)
    _require(not dataset.is_symlink(), "dataset root must not be a symlink")
    context = _capture_context(repository, dataset)
    gate_path = dataset / GATE_PATHS["software_environment_locked"]
    _require(gate_path.is_file(), "software-environment gate template is missing")
    _require(not gate_path.is_symlink(), "software-environment gate must not be a symlink")
    gate = _read_json_mapping(gate_path, name="software-environment gate")
    _validate_gate_template(gate, context)

    lock_source = _ordinary_file(stack_lock_path, name="stack lock")
    wheels = tuple(_ordinary_file(path, name="wheel") for path in wheel_paths)
    lock = load_stack_lock(lock_source)
    verification = verify_stack_lock(lock, wheel_paths=wheels, require_wheels=True)
    _require(verification.get("valid") is True, "stack-lock wheel verification failed")

    lock_entries = {str(entry["name"]): entry for entry in lock["distributions"]}
    wheel_identities = {inspect_wheel(path).name: inspect_wheel(path) for path in wheels}
    _require(set(wheel_identities) == set(STACK_PIPELINE), "wheel set is incomplete")
    freeze_causal4d = context.method_freeze.get("causal4d", {}).get("commit_sha")
    freeze_bpt = context.method_freeze.get("bayesian_phystwin", {}).get("commit_sha")
    _require(
        lock_entries["causal4d"]["source"]["revision"] == freeze_causal4d,
        "stack lock Causal4D revision differs from the method freeze",
    )
    _require(
        lock_entries["bayesian-phystwin"]["source"]["revision"] == freeze_bpt,
        "stack lock BayesianPhysTwin revision differs from the method freeze",
    )

    installed_records = _installed_distribution_records()
    core_stack = _validate_installed_core_stack(lock, installed_records)
    timestamp = _utc_timestamp(completed_at_utc)
    python_record, runtime_record, dependency_report = _runtime_snapshot(
        execution_backend=execution_backend,
        container_image_digest=container_image_digest,
        installed_records=installed_records,
        core_stack=core_stack,
        captured_at_utc=timestamp,
    )

    producer = {
        "name": _safe_nonempty(observation_producer_name, name="observation producer name"),
        "version": _safe_nonempty(
            observation_producer_version,
            name="observation producer version",
        ),
        "artifact_contract": _safe_nonempty(
            observation_artifact_contract,
            name="observation artifact contract",
        ),
    }
    _require(isinstance(prob4d_used, bool), "prob4d_used must be Boolean")
    if prob4d_used:
        _require(
            prob4d_unused_reason is None,
            "a used Prob4D declaration cannot contain an unused reason",
        )
        prob4d_contract = _safe_nonempty(
            prob4d_observation_contract_version,
            name="Prob4D observation contract version",
        )
    else:
        reason = _safe_nonempty(prob4d_unused_reason, name="unused Prob4D reason")
        _require(
            prob4d_observation_contract_version is None,
            "unused Prob4D must not declare an observation contract version",
        )
        prob4d_contract = None

    lock_id = str(lock["lock_id"])
    relative_directory = Path("preacquisition") / "software_environment" / lock_id
    final_directory = dataset / relative_directory
    _require(not final_directory.exists(), "software-environment evidence already exists")
    parent = final_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    _require(not parent.is_symlink(), "software-environment evidence parent is a symlink")
    temporary = Path(tempfile.mkdtemp(prefix=f".{lock_id}.", dir=parent))
    published = False
    try:
        relative_wheels: dict[str, str] = {}
        descriptors: dict[str, dict[str, Any]] = {}
        for name in STACK_PIPELINE:
            identity = wheel_identities[name]
            destination = temporary / identity.filename
            _copy_file(identity.path, destination)
            relative = (relative_directory / identity.filename).as_posix()
            relative_wheels[name] = relative
            descriptors[name] = _descriptor(destination, relative_path=relative)

        normalized_lock_path = temporary / "stack-lock.json"
        atomic_write_json(normalized_lock_path, lock, overwrite=False)
        stack_lock_relative = (relative_directory / "stack-lock.json").as_posix()
        descriptors["stack_lock"] = _descriptor(
            normalized_lock_path,
            relative_path=stack_lock_relative,
        )

        portable_verification = _portable_verification_report(
            verification,
            relative_wheels,
        )
        verification_path = temporary / "stack-verification.json"
        atomic_write_json(verification_path, portable_verification, overwrite=False)
        verification_relative = (
            relative_directory / "stack-verification.json"
        ).as_posix()
        descriptors["stack_verification"] = _descriptor(
            verification_path,
            relative_path=verification_relative,
        )

        dependency_report["stack_lock_id"] = lock_id
        dependency_report["stack_lock"] = stack_lock_relative
        dependency_report["stack_verification"] = verification_relative
        dependency_path = temporary / "resolved-environment.json"
        atomic_write_json(dependency_path, dependency_report, overwrite=False)
        dependency_relative = (
            relative_directory / "resolved-environment.json"
        ).as_posix()
        descriptors["dependency_report"] = _descriptor(
            dependency_path,
            relative_path=dependency_relative,
        )
        runtime_record["resolved_dependency_report"] = dependency_relative

        checks: dict[str, Any] = {
            "method_freeze_sha256": context.method_freeze_sha256,
            "method_freeze_validation_sha256": (
                context.method_freeze_validation_sha256
            ),
            "causal4d": {
                "commit_sha": freeze_causal4d,
                "version": lock_entries["causal4d"]["version"],
                "distribution": descriptors["causal4d"],
            },
            "bayesian_phystwin": {
                "commit_sha": freeze_bpt,
                "version": lock_entries["bayesian-phystwin"]["version"],
                "distribution": descriptors["bayesian-phystwin"],
            },
            "prob4d": (
                {
                    "used": True,
                    "commit_sha": lock_entries["prob4d"]["source"]["revision"],
                    "version": lock_entries["prob4d"]["version"],
                    "observation_contract_version": prob4d_contract,
                    "distribution": descriptors["prob4d"],
                }
                if prob4d_used
                else {"used": False, "reason": reason}
            ),
            "observation_producer": producer,
            "python": python_record,
            "runtime_environment": runtime_record,
            "stack_lock_id": lock_id,
            "stack_lock": stack_lock_relative,
            "stack_verification": verification_relative,
            "capture_schema_name": CAPTURE_SCHEMA_NAME,
            "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        }
        evidence = sorted(descriptors.values(), key=lambda item: str(item["path"]))
        gate["completed_at_utc"] = timestamp
        gate["target_outcomes_used"] = False
        gate["checks"] = checks
        gate["evidence"] = evidence
        gate["locked_before_confirmatory_collection"] = None
        gate["approval"] = {
            "approved": False,
            "approver_id": None,
            "approved_at_utc": None,
        }
        gate["artifact_sha256"] = None

        os.replace(temporary, final_directory)
        published = True
        try:
            atomic_write_json(gate_path, gate, overwrite=True)
        except BaseException:
            shutil.rmtree(final_directory, ignore_errors=True)
            published = False
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)

    _require(published, "software-environment evidence was not published")
    return {
        "passed": True,
        "ready_to_seal": True,
        "approved": False,
        "gate_path": str(gate_path.resolve()),
        "evidence_directory": str(final_directory.resolve()),
        "stack_lock_id": lock_id,
        "prob4d_used": prob4d_used,
        "execution_backend": execution_backend,
        "evidence": evidence,
        "next_command": (
            "causal4d protocol readiness seal-gate "
            f"{repository} {dataset} software_environment_locked "
            "--approved-by <independent-registered-operator>"
        ),
    }


__all__ = [
    "CAPTURE_GENERATOR",
    "CAPTURE_SCHEMA_NAME",
    "CAPTURE_SCHEMA_VERSION",
    "capture_software_environment_template",
]
