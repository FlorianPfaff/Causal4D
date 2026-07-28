"""Fail-closed provenance binding for execution-block calibration folds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


EXECUTION_BLOCK_PROVENANCE_SCHEMA_VERSION = 1
REQUIRED_EXECUTION_BLOCK_BINDING_FIELDS = (
    "protocol_id",
    "protocol_design_sha256",
    "preacquisition_plan_id",
    "preacquisition_amendment_sha256",
)
OPTIONAL_EXECUTION_BLOCK_BINDING_FIELDS = ("method_freeze_sha256",)
_SHA256_FIELDS = {
    "protocol_design_sha256",
    "preacquisition_amendment_sha256",
    "method_freeze_sha256",
    "source_manifest_sha256",
    "target_manifest_sha256",
}


def _require_nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    digest = _require_nonempty_text(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _validated_field(value: Any, *, name: str) -> str:
    if name in _SHA256_FIELDS:
        return _require_sha256(value, name=name)
    return _require_nonempty_text(value, name=name)


def extract_execution_block_manifest_binding(
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Extract the frozen protocol identity required by a calibration manifest."""

    if not isinstance(manifest, Mapping):
        raise ValueError("execution-block manifest root must be a JSON object")
    binding = {
        name: _validated_field(manifest.get(name), name=name)
        for name in REQUIRED_EXECUTION_BLOCK_BINDING_FIELDS
    }
    for name in OPTIONAL_EXECUTION_BLOCK_BINDING_FIELDS:
        if name in manifest:
            binding[name] = _validated_field(manifest[name], name=name)
    return binding


def bind_execution_block_source_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, str]:
    """Bind a source manifest and its bytes to a calibration artifact."""

    return {
        **extract_execution_block_manifest_binding(manifest),
        "source_manifest_sha256": _require_sha256(
            manifest_sha256,
            name="source_manifest_sha256",
        ),
    }


def validate_execution_block_target_manifest(
    calibration_metadata: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    *,
    expected_outer_fold_id: str,
    target_manifest_sha256: str,
) -> dict[str, Any]:
    """Require source and target to share one frozen protocol/amendment identity."""

    expected_fold = _require_nonempty_text(
        expected_outer_fold_id,
        name="expected_outer_fold_id",
    )
    source = extract_execution_block_manifest_binding(calibration_metadata)
    source_manifest_sha256 = _require_sha256(
        calibration_metadata.get("source_manifest_sha256"),
        name="source_manifest_sha256",
    )
    target = extract_execution_block_manifest_binding(target_manifest)
    target_fold = _require_nonempty_text(
        target_manifest.get("outer_fold_id"),
        name="outer_fold_id",
    )
    if target_fold != expected_fold:
        raise ValueError(
            "target outer_fold_id differs from the calibration artifact: "
            f"{target_fold!r} != {expected_fold!r}"
        )

    for name in REQUIRED_EXECUTION_BLOCK_BINDING_FIELDS:
        if target[name] != source[name]:
            raise ValueError(
                f"target {name} differs from the frozen source calibration"
            )

    source_freeze = source.get("method_freeze_sha256")
    target_freeze = target.get("method_freeze_sha256")
    if (source_freeze is None) != (target_freeze is None):
        raise ValueError(
            "method_freeze_sha256 must be present in both source and target "
            "manifests or in neither"
        )
    if source_freeze is not None and target_freeze != source_freeze:
        raise ValueError(
            "target method_freeze_sha256 differs from the frozen source calibration"
        )

    binding: dict[str, Any] = {
        "schema_version": EXECUTION_BLOCK_PROVENANCE_SCHEMA_VERSION,
        "artifact_kind": "ExecutionBlockSourceTargetBinding",
        "verified": True,
        "outer_fold_id": expected_fold,
        **source,
        "source_manifest_sha256": source_manifest_sha256,
        "target_manifest_sha256": _require_sha256(
            target_manifest_sha256,
            name="target_manifest_sha256",
        ),
    }
    return binding
