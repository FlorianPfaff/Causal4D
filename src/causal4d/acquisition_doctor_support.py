"""Shared checks for the fail-closed acquisition doctor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Literal
import uuid

from causal4d.acquisition_flight_common import (
    _assert_json_value,
    _assert_ordinary_file_or_missing,
    _require,
)

CheckStatus = Literal["pass", "warn", "fail", "skipped"]


@dataclass(frozen=True)
class DoctorThresholds:
    minimum_free_bytes: int = 20 * 1024**3
    write_probe_bytes: int = 8 * 1024**2
    minimum_write_mib_s: float = 25.0

    def __post_init__(self) -> None:
        _require(self.minimum_free_bytes >= 0, "minimum_free_bytes must be nonnegative")
        _require(self.write_probe_bytes >= 0, "write_probe_bytes must be nonnegative")
        _require(
            self.minimum_write_mib_s >= 0.0,
            "minimum_write_mib_s must be nonnegative",
        )


def _doctor_check(
    check_id: str,
    status: CheckStatus,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {"check_id": check_id, "status": status, "message": message, **details}


def _measure_write_rate(directory: Path, byte_count: int) -> float:
    if byte_count == 0:
        return 0.0
    path = directory / f".causal4d-write-probe-{uuid.uuid4().hex}.tmp"
    _assert_ordinary_file_or_missing(path, name="write probe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    block = b"\0" * min(1024 * 1024, byte_count)
    written = 0
    start = time.perf_counter()
    descriptor = os.open(path, flags, 0o600)
    try:
        while written < byte_count:
            chunk = block[: min(len(block), byte_count - written)]
            count = os.write(descriptor, chunk)
            _require(count > 0, "storage write probe made no forward progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return (written / 1024**2) / elapsed


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    _assert_ordinary_file_or_missing(path, name=name)
    _require(path.is_file(), f"{name} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), f"{name} must be a JSON object")
    _assert_json_value(payload, name=name)
    return dict(payload)


def _execution_progress_from_status(
    protocol: Mapping[str, Any],
    evidence_status: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    raw_results = evidence_status.get("executions")
    _require(isinstance(raw_results, list), "real evidence status lacks executions")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_results:
        _require(isinstance(raw, Mapping), "execution status must be an object")
        identifier = str(raw.get("execution_id", ""))
        _require(bool(identifier), "execution status lacks execution_id")
        _require(identifier not in by_id, f"duplicate execution status: {identifier}")
        by_id[identifier] = raw

    completed: list[str] = []
    malformed: list[dict[str, Any]] = []
    present: list[str] = []
    executions = sorted(
        protocol.get("executions", []),
        key=lambda item: int(item["acquisition_execution_index"]),
    )
    expected_ids = [str(execution["execution_id"]) for execution in executions]
    _require(
        set(by_id) == set(expected_ids),
        "real evidence status execution set differs from the protocol",
    )
    for identifier in expected_ids:
        result = by_id[identifier]
        manifest_present = result.get("manifest_present") is True
        acquired = result.get("acquired") is True
        validated = result.get("validated") is True
        _require(
            not acquired or manifest_present,
            f"acquired execution lacks a manifest: {identifier}",
        )
        _require(
            not validated or acquired,
            f"validated execution is not acquired: {identifier}",
        )
        if manifest_present:
            present.append(identifier)
        if validated:
            completed.append(identifier)
        elif manifest_present:
            malformed.append(
                {
                    "execution_id": identifier,
                    "acquisition_status": result.get("acquisition_status"),
                    "error": result.get("error"),
                }
            )
    return completed, malformed, present


def _validate_readiness_report(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    from causal4d.preacquisition_readiness_contracts import (
        READINESS_ARTIFACT_KIND,
        READINESS_SCHEMA_VERSION,
        readiness_evidence_sha256,
        readiness_status_sha256,
    )

    values = dict(report)
    _require(
        values.get("schema_version") == READINESS_SCHEMA_VERSION,
        "wrong readiness schema",
    )
    _require(
        values.get("artifact_kind") == READINESS_ARTIFACT_KIND,
        "wrong readiness kind",
    )
    _require(
        values.get("protocol_id") == protocol.get("protocol_id"),
        "readiness protocol mismatch",
    )
    _require(
        values.get("protocol_design_sha256") == protocol.get("design_sha256"),
        "readiness protocol digest mismatch",
    )
    _require(
        values.get("verify_file_hashes") is True,
        "readiness omitted file hash checks",
    )
    _require(values.get("valid") is True, "readiness report is invalid")
    _require(values.get("ready") is True, "readiness report did not pass")
    gate = values.get("collection_gate")
    _require(isinstance(gate, Mapping), "readiness collection gate is missing")
    _require(
        gate.get("first_confirmatory_execution_allowed") is True,
        "readiness did not permit the first confirmatory execution",
    )
    _require(
        values.get("evidence_sha256") == readiness_evidence_sha256(values),
        "readiness portable evidence checksum mismatch",
    )
    _require(
        values.get("status_sha256") == readiness_status_sha256(values),
        "readiness status checksum mismatch",
    )
    return values
