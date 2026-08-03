"""Environment, freeze, readiness, and storage checks for acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil
from typing import Any

from causal4d.acquisition_flight_common import _is_sha256, _require
from causal4d.acquisition_doctor_support import (
    CheckStatus,
    DoctorThresholds,
    _doctor_check,
    _load_json_object,
    _measure_write_rate,
    _validate_readiness_report,
)


def protocol_schedule(
    protocol: Mapping[str, Any],
) -> tuple[str, str, list[Mapping[str, Any]], dict[str, Any]]:
    protocol_id = protocol.get("protocol_id")
    design_sha = protocol.get("design_sha256")
    _require(
        isinstance(protocol_id, str) and bool(protocol_id),
        "protocol_id is missing",
    )
    _require(_is_sha256(design_sha), "protocol design_sha256 is invalid")
    executions = sorted(
        list(protocol.get("executions", [])),
        key=lambda item: int(item["acquisition_execution_index"]),
    )
    _require(executions, "protocol contains no executions")
    expected_indices = list(range(len(executions)))
    actual_indices = [int(item["acquisition_execution_index"]) for item in executions]
    _require(
        actual_indices == expected_indices,
        "protocol execution order is not contiguous",
    )
    check = _doctor_check(
        "protocol_schedule",
        "pass",
        "Protocol execution order is contiguous.",
        execution_count=len(executions),
    )
    return protocol_id, str(design_sha), executions, check


def frozen_checkout_check(
    repository: Path,
    dataset: Path,
    method_freeze_path: str | Path | None,
) -> dict[str, Any]:
    freeze_path = Path(method_freeze_path or dataset / "method_freeze.json")
    try:
        from causal4d.real_experiment_freeze import (
            load_method_freeze_manifest,
            validate_method_freeze_manifest,
            validate_repository_checkout,
        )

        freeze = load_method_freeze_manifest(freeze_path)
        freeze_validation = validate_method_freeze_manifest(
            freeze,
            repository,
            verify_files=True,
        )
        checkout = validate_repository_checkout(freeze, repository)
    except (OSError, KeyError, TypeError, ValueError) as error:
        return _doctor_check(
            "frozen_checkout",
            "fail",
            f"Frozen checkout validation failed: {error}",
            method_freeze_path=str(freeze_path),
        )
    return _doctor_check(
        "frozen_checkout",
        "pass",
        "Checkout and locked files match the method freeze.",
        method_freeze_path=str(freeze_path),
        freeze_validation=freeze_validation,
        checkout=checkout,
    )


def readiness_check(
    protocol: Mapping[str, Any],
    dataset: Path,
    readiness_path: str | Path | None,
) -> dict[str, Any]:
    readiness_file = Path(readiness_path or dataset / "preacquisition-readiness.json")
    try:
        readiness = _validate_readiness_report(
            _load_json_object(readiness_file, name="pre-acquisition readiness report"),
            protocol,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        return _doctor_check(
            "sealed_readiness",
            "fail",
            f"Sealed readiness validation failed: {error}",
            readiness_path=str(readiness_file),
        )
    relocated = Path(str(readiness.get("dataset_root", dataset))) != dataset.resolve()
    return _doctor_check(
        "sealed_readiness",
        "warn" if relocated else "pass",
        (
            "Readiness evidence is valid; dataset path differs because the archive "
            "was relocated."
            if relocated
            else "Readiness evidence is valid and permits confirmatory collection."
        ),
        readiness_path=str(readiness_file),
        evidence_sha256=readiness["evidence_sha256"],
        relocated=relocated,
    )


def storage_checks(
    dataset: Path,
    settings: DoctorThresholds,
    *,
    perform_write_probe: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    usage = shutil.disk_usage(dataset)
    capacity_status: CheckStatus = (
        "pass" if usage.free >= settings.minimum_free_bytes else "fail"
    )
    checks.append(
        _doctor_check(
            "storage_capacity",
            capacity_status,
            (
                "Free storage meets the configured threshold."
                if capacity_status == "pass"
                else "Free storage is below the configured threshold."
            ),
            free_bytes=usage.free,
            minimum_free_bytes=settings.minimum_free_bytes,
        )
    )

    if perform_write_probe and settings.write_probe_bytes > 0:
        try:
            write_rate = _measure_write_rate(dataset, settings.write_probe_bytes)
        except (OSError, ValueError) as error:
            checks.append(
                _doctor_check(
                    "storage_write_probe",
                    "fail",
                    f"Storage write probe failed: {error}",
                    write_probe_bytes=settings.write_probe_bytes,
                )
            )
        else:
            write_status: CheckStatus = (
                "pass" if write_rate >= settings.minimum_write_mib_s else "fail"
            )
            checks.append(
                _doctor_check(
                    "storage_write_probe",
                    write_status,
                    (
                        "Storage write rate meets the configured threshold."
                        if write_status == "pass"
                        else "Storage write rate is below the configured threshold."
                    ),
                    write_mib_s=write_rate,
                    minimum_write_mib_s=settings.minimum_write_mib_s,
                    write_probe_bytes=settings.write_probe_bytes,
                )
            )
    else:
        checks.append(
            _doctor_check(
                "storage_write_probe",
                "skipped",
                "Storage write probe was disabled.",
            )
        )
    return checks
