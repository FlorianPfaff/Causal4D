"""Readiness prerequisite for the exact registered real-analysis manifest."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.real_evidence_common import _error_text, _parse_utc_timestamp
from causal4d.registered_real_analysis import (
    validate_registered_real_analysis_manifest,
)

REGISTERED_ANALYSIS_PATH = "registered-analysis.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_registered_real_analysis_prerequisite(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    method_freeze_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact analysis registration needed to authorize collection."""

    root = Path(dataset_root)
    path = root / REGISTERED_ANALYSIS_PATH
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "present": path.is_file(),
        "template": False,
        "valid": False,
        "error": None,
    }
    if not result["present"]:
        result["error"] = f"{REGISTERED_ANALYSIS_PATH} is missing"
        return result

    try:
        _require(
            method_freeze_result.get("valid") is True,
            "registered analysis requires a valid method freeze",
        )
        expected_freeze_sha = method_freeze_result.get("sha256")
        _require(
            isinstance(expected_freeze_sha, str) and len(expected_freeze_sha) == 64,
            "method-freeze SHA-256 is unavailable",
        )
        freeze_path = root / "method_freeze.json"
        freeze_snapshot = read_regular_file(freeze_path, name="method freeze")
        _require(
            freeze_snapshot.sha256 == expected_freeze_sha,
            "registered analysis prerequisite sees a different method freeze",
        )
        freeze = load_strict_json_object(
            freeze_snapshot.payload,
            name="method freeze",
        )
        analysis_snapshot = read_regular_file(
            path,
            name="registered analysis manifest",
        )
        analysis = load_strict_json_object(
            analysis_snapshot.payload,
            name="registered analysis manifest",
        )
        preacquisition = freeze.get("preacquisition")
        _require(
            isinstance(preacquisition, Mapping),
            "method freeze lacks pre-acquisition provenance",
        )
        validated = validate_registered_real_analysis_manifest(
            analysis,
            expected_protocol_id=str(protocol["protocol_id"]),
            expected_protocol_design_sha256=str(protocol["design_sha256"]),
            expected_preacquisition_amendment_sha256=str(
                preacquisition["amendment_sha256"]
            ),
            expected_method_freeze_sha256=expected_freeze_sha,
        )
        causal4d = freeze.get("causal4d")
        bpt = freeze.get("bayesian_phystwin")
        _require(
            isinstance(causal4d, Mapping) and isinstance(bpt, Mapping),
            "method freeze lacks software provenance",
        )
        software = cast(Mapping[str, Any], validated["software"])
        _require(
            software["causal4d_commit_sha"] == causal4d.get("commit_sha"),
            "registered analysis binds a different Causal4D commit",
        )
        _require(
            software["bayesian_phystwin_commit_sha"] == bpt.get("commit_sha"),
            "registered analysis binds a different BayesianPhysTwin commit",
        )
        registered_at = _parse_utc_timestamp(
            validated["registered_at_utc"],
            name="registered analysis registered_at_utc",
        )
        frozen_at = _parse_utc_timestamp(
            freeze.get("frozen_at_utc"),
            name="method freeze frozen_at_utc",
        )
        _require(
            registered_at >= frozen_at,
            "registered analysis predates the method freeze",
        )
        result.update(
            {
                "valid": True,
                "analysis_id": validated["analysis_id"],
                "registered_at_utc": validated["registered_at_utc"],
                "registered_by": validated["registered_by"],
                "method_freeze_sha256": expected_freeze_sha,
                "sha256": analysis_snapshot.sha256,
                "bytes": analysis_snapshot.byte_count,
                "target_outcomes_used": False,
            }
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
    return result


__all__ = [
    "REGISTERED_ANALYSIS_PATH",
    "validate_registered_real_analysis_prerequisite",
]
