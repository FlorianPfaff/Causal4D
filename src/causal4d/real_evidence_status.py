"""Fail-closed progress and claim-readiness reporting for real evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from causal4d.real_protocol import (
    validate_execution_manifest,
    validate_object_registration,
    validate_protocol,
    validate_slip_pilot,
    write_acquisition_schedule,
)

EVIDENCE_STATUS_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return dict(payload)


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _prerequisite_result(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.is_file(),
        "valid": False,
        "error": None,
    }


def _validate_dataset_protocol(
    protocol: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    result = _prerequisite_result(path)
    if not result["present"]:
        result["error"] = "dataset protocol.json is missing"
        return result
    try:
        candidate = _load_json_mapping(path)
        validate_protocol(candidate)
        if candidate != dict(protocol):
            raise ValueError("dataset protocol differs from the locked protocol")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    result["valid"] = True
    result["sha256"], result["bytes"] = _sha256_file(path)
    return result


def _expected_schedule_rows(protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="causal4d-status-") as directory:
        generated = write_acquisition_schedule(
            Path(directory) / "acquisition_schedule.csv",
            protocol,
        )
        with generated.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def _validate_acquisition_schedule(
    protocol: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    result = _prerequisite_result(path)
    if not result["present"]:
        result["error"] = "acquisition_schedule.csv is missing"
        return result
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        expected = _expected_schedule_rows(protocol)
        if rows != expected:
            raise ValueError("acquisition schedule differs from the locked design")
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    result["valid"] = True
    result["row_count"] = len(rows)
    result["sha256"], result["bytes"] = _sha256_file(path)
    return result


def _validate_registration_files(
    dataset_root: Path,
    registration: Mapping[str, Any],
) -> None:
    for region_id, descriptor in registration["contact_regions"].items():
        relative = Path(descriptor["canonical_node_set_path"])
        node_path = dataset_root / relative
        if not node_path.is_file():
            raise ValueError(f"contact node set is missing: {region_id}")
        digest, _ = _sha256_file(node_path)
        if digest != descriptor["canonical_node_set_sha256"]:
            raise ValueError(f"contact node-set checksum mismatch: {region_id}")


def _validate_object_registration_prerequisite(
    protocol: Mapping[str, Any],
    dataset_root: Path,
    path: Path,
    *,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    result = _prerequisite_result(path)
    result["file_hashes_verified"] = None if not verify_file_hashes else False
    if not result["present"]:
        result["error"] = "object_registration.json is missing"
        return result
    try:
        registration = _load_json_mapping(path)
        validate_object_registration(protocol, registration)
        if verify_file_hashes:
            _validate_registration_files(dataset_root, registration)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    result["valid"] = True
    result["file_hashes_verified"] = True if verify_file_hashes else None
    result["sha256"], result["bytes"] = _sha256_file(path)
    return result


def _validate_json_prerequisite(
    protocol: Mapping[str, Any],
    path: Path,
    validator: Callable[[Mapping[str, Any], Mapping[str, Any]], None],
    *,
    missing_message: str,
) -> dict[str, Any]:
    result = _prerequisite_result(path)
    if not result["present"]:
        result["error"] = missing_message
        return result
    try:
        payload = _load_json_mapping(path)
        validator(protocol, payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    result["valid"] = True
    result["sha256"], result["bytes"] = _sha256_file(path)
    return result


def _execution_status(
    protocol: Mapping[str, Any],
    dataset_root: Path,
    execution: Mapping[str, Any],
    *,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    execution_id = str(execution["execution_id"])
    execution_root = dataset_root / "executions" / execution_id
    manifest_path = execution_root / "manifest.json"
    template_path = execution_root / "manifest.template.json"
    result: dict[str, Any] = {
        "execution_id": execution_id,
        "session_id": str(execution["session_id"]),
        "acquisition_execution_index": int(execution["acquisition_execution_index"]),
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_path.is_file(),
        "manifest_parsed": False,
        "template_present": template_path.is_file(),
        "acquisition_status": None,
        "acquired": False,
        "validated": False,
        "included": None,
        "quality_gate_failures": [],
        "file_hashes_verified": None if not verify_file_hashes else False,
        "error": None,
    }
    if not result["manifest_present"]:
        result["error"] = "manifest.json is missing"
        return result
    try:
        manifest = _load_json_mapping(manifest_path)
        result["manifest_parsed"] = True
        result["acquisition_status"] = manifest.get("acquisition_status")
        result["acquired"] = result["acquisition_status"] == "complete"
        result["manifest_sha256"], result["manifest_bytes"] = _sha256_file(
            manifest_path
        )
        if not result["acquired"]:
            raise ValueError("execution manifest is not explicitly complete")
        validation = validate_execution_manifest(
            protocol,
            manifest,
            execution_root=execution_root,
            verify_files=verify_file_hashes,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    result["validated"] = True
    result["included"] = bool(validation["included"])
    result["quality_gate_failures"] = list(validation["quality_gate_failures"])
    result["file_hashes_verified"] = True if verify_file_hashes else None
    return result


def _unexpected_execution_directories(
    dataset_root: Path,
    expected_ids: set[str],
) -> list[str]:
    root = dataset_root / "executions"
    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name not in expected_ids
    )


def _blockers(
    *,
    prerequisites: Mapping[str, Mapping[str, Any]],
    execution_results: list[Mapping[str, Any]],
    unexpected_execution_directories: list[str],
    verify_file_hashes: bool,
) -> list[str]:
    blockers = [
        f"prerequisite:{name}"
        for name, result in prerequisites.items()
        if not result["valid"]
    ]
    missing = [
        str(result["execution_id"])
        for result in execution_results
        if not result["manifest_present"]
    ]
    incomplete = [
        str(result["execution_id"])
        for result in execution_results
        if result["manifest_parsed"] and not result["acquired"]
    ]
    invalid = [
        str(result["execution_id"])
        for result in execution_results
        if result["manifest_present"]
        and (
            not result["manifest_parsed"]
            or (result["acquired"] and not result["validated"])
        )
    ]
    if missing:
        blockers.append(f"missing_execution_manifests:{len(missing)}")
    if incomplete:
        blockers.append(f"incomplete_execution_manifests:{len(incomplete)}")
    if invalid:
        blockers.append(f"invalid_execution_manifests:{len(invalid)}")
    if unexpected_execution_directories:
        blockers.append(
            f"unexpected_execution_directories:{len(unexpected_execution_directories)}"
        )
    if not verify_file_hashes:
        blockers.append("file_hashes_not_verified")
    return blockers


def build_real_evidence_status(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    """Summarize progress without treating scaffolded templates as evidence."""

    validate_protocol(protocol)
    root = Path(dataset_root)
    prerequisites = {
        "dataset_protocol": _validate_dataset_protocol(
            protocol,
            root / "protocol.json",
        ),
        "acquisition_schedule": _validate_acquisition_schedule(
            protocol,
            root / "acquisition_schedule.csv",
        ),
        "object_registration": _validate_object_registration_prerequisite(
            protocol,
            root,
            root / "object_registration.json",
            verify_file_hashes=verify_file_hashes,
        ),
        "slip_pilot": _validate_json_prerequisite(
            protocol,
            root / "slip_pilot.json",
            validate_slip_pilot,
            missing_message="slip_pilot.json is missing",
        ),
    }
    executions = sorted(
        protocol["executions"],
        key=lambda value: int(value["acquisition_execution_index"]),
    )
    execution_results = [
        _execution_status(
            protocol,
            root,
            execution,
            verify_file_hashes=verify_file_hashes,
        )
        for execution in executions
    ]
    expected_ids = {str(execution["execution_id"]) for execution in executions}
    unexpected = _unexpected_execution_directories(root, expected_ids)
    specified = len(execution_results)
    manifest_count = sum(
        bool(result["manifest_present"]) for result in execution_results
    )
    acquired = sum(bool(result["acquired"]) for result in execution_results)
    validated = sum(bool(result["validated"]) for result in execution_results)
    included = sum(result["included"] is True for result in execution_results)
    excluded = sum(result["included"] is False for result in execution_results)
    prerequisites_valid = all(result["valid"] for result in prerequisites.values())
    accounting_complete = included + excluded == specified
    complete = bool(
        prerequisites_valid
        and manifest_count == specified
        and acquired == specified
        and validated == specified
        and accounting_complete
        and not unexpected
    )
    file_hashes_verified = bool(verify_file_hashes and complete)
    claim_ready = bool(complete and file_hashes_verified)
    blockers = _blockers(
        prerequisites=prerequisites,
        execution_results=execution_results,
        unexpected_execution_directories=unexpected,
        verify_file_hashes=verify_file_hashes,
    )
    if complete and verify_file_hashes:
        blockers = []
    next_pending = next(
        (
            {
                "execution_id": result["execution_id"],
                "session_id": result["session_id"],
                "acquisition_execution_index": result["acquisition_execution_index"],
            }
            for result in execution_results
            if not result["validated"]
        ),
        None,
    )
    return {
        "schema_version": EVIDENCE_STATUS_SCHEMA_VERSION,
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_design_sha256": str(protocol["design_sha256"]),
        "dataset_root": str(root.resolve()),
        "specified_executions": specified,
        "manifest_executions": manifest_count,
        "acquired_executions": acquired,
        "validated_executions": validated,
        "included_executions": included,
        "excluded_executions": excluded,
        "missing_execution_ids": [
            result["execution_id"]
            for result in execution_results
            if not result["manifest_present"]
        ],
        "incomplete_execution_ids": [
            result["execution_id"]
            for result in execution_results
            if result["manifest_parsed"] and not result["acquired"]
        ],
        "invalid_execution_ids": [
            result["execution_id"]
            for result in execution_results
            if result["manifest_present"]
            and (
                not result["manifest_parsed"]
                or (result["acquired"] and not result["validated"])
            )
        ],
        "unexpected_execution_directories": unexpected,
        "next_pending_execution": next_pending,
        "prerequisites": prerequisites,
        "executions": execution_results,
        "file_hashes_requested": verify_file_hashes,
        "file_hashes_verified": file_hashes_verified,
        "accounting_complete": accounting_complete,
        "complete": complete,
        "claim_ready": claim_ready,
        "passed": claim_ready,
        "blockers": blockers,
    }


def write_real_evidence_status(
    path: str | Path,
    status: Mapping[str, Any],
) -> Path:
    """Atomically write one deterministic, human-readable status snapshot."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(status),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor: int | None = None
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output


__all__ = [
    "EVIDENCE_STATUS_SCHEMA_VERSION",
    "build_real_evidence_status",
    "write_real_evidence_status",
]
