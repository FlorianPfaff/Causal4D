"""Method-freeze evidence and independent attestation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.real_evidence_common import (
    METHOD_FREEZE_ATTESTATION_SCHEMA_VERSION,
    _error_text,
    _finalize_prerequisite,
    _load_json_mapping,
    _parse_utc_timestamp,
    _prerequisite_result,
    _require,
    _sha256_file,
)
from causal4d.real_experiment_freeze import (
    validate_method_freeze_manifest,
    validate_repository_checkout,
)
from causal4d.real_protocol import validate_protocol


def method_freeze_validation_attestation_template(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an explicitly incomplete independent freeze-validation record."""

    validate_protocol(protocol)
    return {
        "schema_version": METHOD_FREEZE_ATTESTATION_SCHEMA_VERSION,
        "artifact_kind": "MethodFreezeValidationAttestation",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "method_freeze_sha256": None,
        "causal4d_commit_sha": None,
        "bayesian_phystwin_commit_sha": None,
        "verifier_id": None,
        "verified_at_utc": None,
        "independent_of_freezer": None,
        "repository_checkout_verified": None,
        "locked_file_hashes_verified": None,
        "validation_passed": None,
    }


def build_method_freeze_validation_attestation(
    protocol: Mapping[str, Any],
    method_freeze_path: str | Path,
    repository_root: str | Path,
    *,
    verified_by: str,
    verified_at_utc: str | None = None,
) -> dict[str, Any]:
    """Independently validate and bind an exact method-freeze file."""

    validate_protocol(protocol)
    path = Path(method_freeze_path)
    method_freeze = _load_json_mapping(path)
    verifier_id = str(verified_by).strip()
    _require(bool(verifier_id), "verified_by is required")
    freezer_id = method_freeze.get("frozen_by")
    _require(
        not isinstance(freezer_id, str)
        or verifier_id.casefold() != freezer_id.strip().casefold(),
        "method freeze must be independently verified",
    )
    commit_sha = method_freeze.get("causal4d", {}).get("commit_sha")
    validation = validate_method_freeze_manifest(
        method_freeze,
        repository_root,
        expected_causal4d_commit_sha=commit_sha,
        verify_files=True,
    )
    checkout = validate_repository_checkout(method_freeze, repository_root)
    _require(not checkout["dirty_worktree"], "acquisition checkout is dirty")
    timestamp_text = verified_at_utc or datetime.now(timezone.utc).isoformat()
    verified_at = _parse_utc_timestamp(
        timestamp_text,
        name="freeze verified_at_utc",
    )
    frozen_at = _parse_utc_timestamp(
        method_freeze.get("frozen_at_utc"),
        name="freeze frozen_at_utc",
    )
    _require(verified_at >= frozen_at, "freeze attestation predates the method freeze")
    freeze_sha256, _ = _sha256_file(path)
    return {
        "schema_version": METHOD_FREEZE_ATTESTATION_SCHEMA_VERSION,
        "artifact_kind": "MethodFreezeValidationAttestation",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "method_freeze_sha256": freeze_sha256,
        "causal4d_commit_sha": validation["causal4d_commit_sha"],
        "bayesian_phystwin_commit_sha": validation["bayesian_phystwin_commit_sha"],
        "verifier_id": verifier_id,
        "verified_at_utc": timestamp_text,
        "independent_of_freezer": True,
        "repository_checkout_verified": True,
        "locked_file_hashes_verified": bool(validation["file_hashes_verified"]),
        "validation_passed": bool(validation["passed"]),
    }


def write_method_freeze_validation_attestation(
    path: str | Path,
    attestation: Mapping[str, Any],
) -> Path:
    """Write a deterministic method-freeze validation attestation."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(attestation), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _validate_method_freeze_attestation(
    protocol: Mapping[str, Any],
    attestation: Mapping[str, Any],
    *,
    method_freeze: Mapping[str, Any],
    method_freeze_sha256: str,
) -> dict[str, Any]:
    _require(
        attestation.get("schema_version") == METHOD_FREEZE_ATTESTATION_SCHEMA_VERSION,
        "unsupported method-freeze attestation schema",
    )
    _require(
        attestation.get("artifact_kind") == "MethodFreezeValidationAttestation",
        "unexpected method-freeze attestation kind",
    )
    _require(
        attestation.get("protocol_id") == protocol["protocol_id"],
        "attestation protocol mismatch",
    )
    _require(
        attestation.get("protocol_design_sha256") == protocol["design_sha256"],
        "attestation protocol digest mismatch",
    )
    _require(
        attestation.get("method_freeze_sha256") == method_freeze_sha256,
        "attestation binds a different method freeze",
    )
    _require(
        attestation.get("causal4d_commit_sha")
        == method_freeze.get("causal4d", {}).get("commit_sha"),
        "attestation binds a different Causal4D commit",
    )
    _require(
        attestation.get("bayesian_phystwin_commit_sha")
        == method_freeze.get("bayesian_phystwin", {}).get("commit_sha"),
        "attestation binds a different Bayesian-PhysTwin commit",
    )
    verifier_id = attestation.get("verifier_id")
    _require(
        isinstance(verifier_id, str) and bool(verifier_id.strip()),
        "freeze verifier id is missing",
    )
    freezer_id = method_freeze.get("frozen_by")
    _require(
        not isinstance(freezer_id, str)
        or verifier_id.strip().casefold() != freezer_id.strip().casefold(),
        "method freeze must be independently verified",
    )
    verified_at = _parse_utc_timestamp(
        attestation.get("verified_at_utc"), name="freeze verified_at_utc"
    )
    frozen_at = _parse_utc_timestamp(
        method_freeze.get("frozen_at_utc"), name="freeze frozen_at_utc"
    )
    _require(verified_at >= frozen_at, "freeze attestation predates the method freeze")
    for field in (
        "independent_of_freezer",
        "repository_checkout_verified",
        "locked_file_hashes_verified",
        "validation_passed",
    ):
        _require(
            attestation.get(field) is True, f"freeze attestation gate failed: {field}"
        )
    return {
        "passed": True,
        "verifier_id": verifier_id.strip(),
        "verified_at_utc": attestation["verified_at_utc"],
    }


def _validate_method_freeze_prerequisites(
    protocol: Mapping[str, Any],
    dataset_root: Path,
    *,
    repository_root: str | Path | None,
    verify_file_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    freeze_path = dataset_root / "method_freeze.json"
    attestation_path = dataset_root / "method_freeze_validation.json"
    freeze_result = _prerequisite_result(freeze_path)
    attestation_result = _prerequisite_result(attestation_path)
    freeze_result["repository_checkout_verified"] = False
    freeze_result["file_hashes_verified"] = None if not verify_file_hashes else False
    method_freeze: dict[str, Any] | None = None

    if not freeze_result["present"]:
        freeze_result["error"] = "method_freeze.json is missing"
    else:
        try:
            method_freeze = _load_json_mapping(freeze_path)
            _require(
                repository_root is not None,
                "repository_root is required to verify the method freeze",
            )
            validation = validate_method_freeze_manifest(
                method_freeze,
                repository_root,
                expected_causal4d_commit_sha=method_freeze.get("causal4d", {}).get(
                    "commit_sha"
                ),
                verify_files=verify_file_hashes,
            )
            checkout = validate_repository_checkout(method_freeze, repository_root)
            freeze_result.update(validation)
            freeze_result["repository_checkout_verified"] = not checkout[
                "dirty_worktree"
            ]
            freeze_result["file_hashes_verified"] = True if verify_file_hashes else None
            freeze_result["frozen_at_utc"] = method_freeze["frozen_at_utc"]
            freeze_result["valid"] = True
            freeze_result["sha256"], freeze_result["bytes"] = _sha256_file(freeze_path)
        except (OSError, KeyError, TypeError, ValueError) as error:
            freeze_result["error"] = _error_text(error)

    if not attestation_result["present"]:
        attestation_result["error"] = "method_freeze_validation.json is missing"
    else:
        try:
            _require(
                method_freeze is not None, "method_freeze.json must be readable first"
            )
            freeze_sha256, _ = _sha256_file(freeze_path)
            attestation = _load_json_mapping(attestation_path)
            attestation_result.update(
                _validate_method_freeze_attestation(
                    protocol,
                    attestation,
                    method_freeze=method_freeze,
                    method_freeze_sha256=freeze_sha256,
                )
            )
            _finalize_prerequisite(attestation_result, attestation_path)
        except (OSError, KeyError, TypeError, ValueError) as error:
            attestation_result["error"] = _error_text(error)
    return freeze_result, attestation_result, method_freeze


__all__ = [
    "_validate_method_freeze_attestation",
    "_validate_method_freeze_prerequisites",
    "build_method_freeze_validation_attestation",
    "method_freeze_validation_attestation_template",
    "write_method_freeze_validation_attestation",
]
