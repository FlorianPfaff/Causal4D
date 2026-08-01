"""Verify the concrete source artifacts bound by a real-result gate summary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from causal4d.real_experiment_freeze import MILESTONE_ID, SCHEMA_VERSION
from causal4d.real_result_interpretation import RealResultGateSummary

REGISTERED_ANALYSIS_SCHEMA_VERSION: Final = 1
REGISTERED_ANALYSIS_ARTIFACT_KIND: Final = (
    "Causal4DRegisteredRealAnalysisManifest"
)
SOURCE_VERIFICATION_SCHEMA_VERSION: Final = 1
SOURCE_VERIFICATION_ARTIFACT_KIND: Final = (
    "Causal4DRealResultSourceVerification"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(values: Mapping[str, Any], *, omitted_field: str) -> str:
    payload = dict(values)
    payload.pop(omitted_field, None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_descriptor(path: str | Path, *, name: str) -> dict[str, object]:
    source = Path(path)
    _require(not source.is_symlink(), f"{name} must not be a symlink")
    _require(source.is_file(), f"{name} is not a regular file")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
    except OSError as error:
        raise ValueError(f"cannot read {name}") from error
    return {
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
    }


def _read_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be readable JSON") from error
    _require(isinstance(payload, Mapping), f"{name} must contain a JSON object")
    return dict(payload)


def _validate_method_freeze(
    payload: Mapping[str, Any],
    gates: RealResultGateSummary,
) -> None:
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        "method freeze uses an unsupported schema",
    )
    _require(
        payload.get("milestone_id") == MILESTONE_ID,
        "method freeze targets another milestone",
    )
    _require(payload.get("status") == "sealed", "method freeze is not sealed")
    _require(
        payload.get("locked_before_confirmatory_collection") is True,
        "method freeze was not locked before confirmatory collection",
    )
    _require(
        payload.get("target_outcomes_observed_at_freeze") is False,
        "method freeze records target access before locking",
    )
    protocol = payload.get("protocol")
    _require(isinstance(protocol, Mapping), "method freeze lacks protocol provenance")
    _require(
        protocol.get("design_sha256") == gates.protocol_design_sha256,
        "method freeze protocol digest differs from the gate summary",
    )
    preacquisition = payload.get("preacquisition")
    _require(
        isinstance(preacquisition, Mapping),
        "method freeze lacks pre-acquisition provenance",
    )
    _require(
        preacquisition.get("amendment_sha256")
        == gates.preacquisition_amendment_sha256,
        "method freeze amendment digest differs from the gate summary",
    )
    analysis = payload.get("analysis_contract")
    _require(isinstance(analysis, Mapping), "method freeze lacks an analysis contract")
    _require(
        analysis.get("target_outcomes_may_select_method_or_hyperparameters")
        is False,
        "method freeze permits target-informed analysis selection",
    )
    _require(
        analysis.get("optional_branches_may_change_primary_analysis") is False,
        "method freeze permits optional branches to change the primary analysis",
    )


def _validate_registered_analysis(
    payload: Mapping[str, Any],
    gates: RealResultGateSummary,
) -> None:
    _require(
        payload.get("schema_version") == REGISTERED_ANALYSIS_SCHEMA_VERSION,
        "registered analysis manifest uses an unsupported schema",
    )
    _require(
        payload.get("artifact_kind") == REGISTERED_ANALYSIS_ARTIFACT_KIND,
        "unexpected registered analysis artifact kind",
    )
    analysis_id = payload.get("analysis_id")
    _require(
        isinstance(analysis_id, str) and bool(analysis_id.strip()),
        "registered analysis manifest lacks analysis_id",
    )
    for field, expected in (
        ("protocol_id", gates.protocol_id),
        ("protocol_design_sha256", gates.protocol_design_sha256),
        (
            "preacquisition_amendment_sha256",
            gates.preacquisition_amendment_sha256,
        ),
        ("method_freeze_sha256", gates.method_freeze_sha256),
    ):
        _require(
            payload.get(field) == expected,
            f"registered analysis manifest {field} differs from the gate summary",
        )
    _require(
        payload.get("primary_analysis_locked") is True,
        "registered analysis manifest does not lock the primary analysis",
    )
    _require(
        payload.get("target_outcomes_may_select_method_or_hyperparameters")
        is False,
        "registered analysis manifest permits target-informed selection",
    )
    _require(
        payload.get("optional_branches_may_change_primary_analysis") is False,
        "registered analysis manifest permits optional-branch rescue",
    )


def verify_real_result_sources(
    gates: RealResultGateSummary,
    *,
    method_freeze_path: str | Path,
    analysis_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify exact files and their protocol bindings before interpretation."""

    method_descriptor = _artifact_descriptor(
        method_freeze_path,
        name="method freeze",
    )
    _require(
        method_descriptor["sha256"] == gates.method_freeze_sha256,
        "method-freeze file SHA-256 differs from the gate summary",
    )
    method_payload = _read_json_object(
        method_freeze_path,
        name="method freeze",
    )
    _validate_method_freeze(method_payload, gates)

    analysis_descriptor = _artifact_descriptor(
        analysis_manifest_path,
        name="registered analysis manifest",
    )
    _require(
        analysis_descriptor["sha256"] == gates.analysis_manifest_sha256,
        "analysis-manifest file SHA-256 differs from the gate summary",
    )
    analysis_payload = _read_json_object(
        analysis_manifest_path,
        name="registered analysis manifest",
    )
    _validate_registered_analysis(analysis_payload, gates)

    result: dict[str, Any] = {
        "schema_version": SOURCE_VERIFICATION_SCHEMA_VERSION,
        "artifact_kind": SOURCE_VERIFICATION_ARTIFACT_KIND,
        "protocol_id": gates.protocol_id,
        "protocol_design_sha256": gates.protocol_design_sha256,
        "preacquisition_amendment_sha256": (
            gates.preacquisition_amendment_sha256
        ),
        "method_freeze": method_descriptor,
        "registered_analysis_manifest": analysis_descriptor,
    }
    result["verification_sha256"] = _canonical_sha256(
        result,
        omitted_field="verification_sha256",
    )
    return result


def validate_real_result_source_verification(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the portable source-verification record without reopening files."""

    _require(
        payload.get("schema_version") == SOURCE_VERIFICATION_SCHEMA_VERSION,
        "unsupported source-verification schema",
    )
    _require(
        payload.get("artifact_kind") == SOURCE_VERIFICATION_ARTIFACT_KIND,
        "unexpected source-verification artifact kind",
    )
    for field in ("protocol_id", "protocol_design_sha256", "preacquisition_amendment_sha256"):
        value = payload.get(field)
        _require(isinstance(value, str) and bool(value), f"{field} is missing")
    for field in ("method_freeze", "registered_analysis_manifest"):
        descriptor = payload.get(field)
        _require(isinstance(descriptor, Mapping), f"{field} descriptor is missing")
        digest = descriptor.get("sha256")
        byte_count = descriptor.get("bytes")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"{field} SHA-256 is invalid",
        )
        _require(
            isinstance(byte_count, int)
            and not isinstance(byte_count, bool)
            and byte_count > 0,
            f"{field} byte count is invalid",
        )
    expected = _canonical_sha256(payload, omitted_field="verification_sha256")
    _require(
        payload.get("verification_sha256") == expected,
        "source-verification SHA-256 mismatch",
    )
    return dict(payload)


__all__ = [
    "REGISTERED_ANALYSIS_ARTIFACT_KIND",
    "REGISTERED_ANALYSIS_SCHEMA_VERSION",
    "SOURCE_VERIFICATION_ARTIFACT_KIND",
    "SOURCE_VERIFICATION_SCHEMA_VERSION",
    "validate_real_result_source_verification",
    "verify_real_result_sources",
]
