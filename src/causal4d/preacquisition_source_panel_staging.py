"""Read-only preflight validation for a staged physical source manifest."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from causal4d.acquisition_flight_common import (
    _assert_no_symlink_components,
    _assert_ordinary_file_or_missing,
    _reject_target_outcomes,
    _require,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.preacquisition_readiness_contracts import (
    SOURCE_PANEL_MANIFEST_PATH,
    _canonical_sha256,
    _read_json_mapping,
    _resolved_dataset_file,
    _safe_relative_path,
    _sha256_file,
    load_registered_preacquisition_chain,
    source_panel_execution_manifest_template,
)
from causal4d.preacquisition_source_panel_control import build_source_panel_status
from causal4d.preacquisition_source_validation import (
    _validate_source_execution_manifest,
)

SOURCE_PANEL_STAGING_PREFLIGHT_SCHEMA_VERSION = 1
SOURCE_PANEL_STAGING_PREFLIGHT_ARTIFACT_KIND = "Causal4DSourcePanelStagingPreflight"


def source_panel_staging_evidence_sha256(values: Mapping[str, Any]) -> str:
    """Return a mount-independent digest of one staging preflight result."""

    payload = deepcopy(dict(values))
    for field in (
        "repository_root",
        "dataset_root",
        "source_json",
        "source_panel_status_sha256_before",
        "evidence_sha256",
        "status_sha256",
    ):
        payload.pop(field, None)
    command = payload.get("publication_command_argv")
    if isinstance(command, list) and len(command) == 7:
        normalized_command = list(command)
        normalized_command[4] = "${REPOSITORY_ROOT}"
        normalized_command[5] = "${DATASET_ROOT}"
        normalized_command[6] = "${DATASET_ROOT}/" + str(
            values["source_manifest_relative_path"]
        )
        payload["publication_command_argv"] = normalized_command
    payload.pop("publication_command_text", None)
    return _canonical_sha256(payload, omitted_field="evidence_sha256")


def source_panel_staging_status_sha256(values: Mapping[str, Any]) -> str:
    """Return the digest of the exact host-local staging preflight result."""

    return _canonical_sha256(values, omitted_field="status_sha256")


def _resolved_dataset_root(dataset_root: str | Path) -> Path:
    candidate = Path(dataset_root)
    _assert_no_symlink_components(candidate, name="dataset root")
    _require(candidate.is_dir(), "dataset root must exist")
    return candidate.resolve()


def _resolved_staging_file(
    dataset_root: Path,
    source_json: str | Path,
    *,
    execution_id: str,
) -> tuple[Path, str]:
    source = Path(source_json)
    _assert_no_symlink_components(source, name="source-panel staging manifest")
    _require(source.is_file(), "source-panel staging manifest is missing")
    resolved = source.resolve(strict=True)
    try:
        relative = resolved.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(
            "source-panel staging manifest must be directly below dataset_root/staging"
        ) from error
    _require(
        relative.parent == Path("staging"),
        "source-panel staging manifest must be directly below dataset_root/staging",
    )
    _require(
        relative.name == f"{execution_id}.json",
        "source-panel staging filename must match the next execution id",
    )
    staging_root = dataset_root / "staging"
    _assert_no_symlink_components(staging_root, name="source-panel staging directory")
    _require(staging_root.is_dir(), "source-panel staging directory is missing")
    return resolved, relative.as_posix()


def _artifact_snapshot(
    dataset_root: Path,
    artifacts: Any,
    *,
    verify_declared: bool,
) -> list[dict[str, Any]]:
    _require(isinstance(artifacts, list), "source execution artifacts are invalid")
    snapshots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, descriptor in enumerate(artifacts):
        _require(
            isinstance(descriptor, Mapping),
            f"source execution artifact {index} is invalid",
        )
        relative = _safe_relative_path(
            descriptor.get("path"),
            name=f"source execution artifact {index}",
        )
        relative_text = relative.as_posix()
        _require(
            relative_text not in seen,
            f"source execution artifacts contain a duplicate path: {relative_text}",
        )
        path = _resolved_dataset_file(
            dataset_root,
            relative,
            name=f"source execution artifact {index}",
        )
        digest, byte_count = _sha256_file(path)
        if verify_declared:
            _require(
                descriptor.get("sha256") == digest,
                f"source execution artifact {index} checksum mismatch",
            )
            declared_bytes = descriptor.get("bytes")
            _require(
                type(declared_bytes) is int and declared_bytes == byte_count,
                f"source execution artifact {index} byte count mismatch",
            )
        seen.add(relative_text)
        snapshots.append(
            {
                "path": relative_text,
                "sha256": digest,
                "bytes": byte_count,
            }
        )
    _require(bool(snapshots), "source execution artifacts must be nonempty")
    return snapshots


def verify_source_panel_manifest_staging(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
) -> dict[str, Any]:
    """Hash-verify exactly the next staged source manifest without publication."""

    protocol, _, _, v4 = load_registered_preacquisition_chain(repository_root)
    root = _resolved_dataset_root(dataset_root)
    status_before = build_source_panel_status(
        repository_root,
        root,
        verify_file_hashes=True,
    )
    _require(status_before["valid"] is True, "source-panel status is invalid")
    _require(status_before["complete"] is False, "source panel is already complete")
    next_execution = status_before.get("next_execution")
    _require(isinstance(next_execution, Mapping), "source panel has no next execution")
    _require(
        next_execution.get("template_present") is True
        and next_execution.get("template_valid") is True,
        "next source-panel manifest template is missing or invalid",
    )

    execution_id = str(next_execution["execution_id"])
    session_id = str(next_execution["session_id"])
    source, source_relative = _resolved_staging_file(
        root,
        source_json,
        execution_id=execution_id,
    )
    source_digest_before, source_bytes_before = _sha256_file(source)
    payload = _read_json_mapping(source, name="source-panel staging manifest")
    _reject_target_outcomes(payload)
    expected_template = source_panel_execution_manifest_template(
        next_execution,
        protocol,
        v4,
    )
    _require(
        set(payload) == set(expected_template),
        "source-panel manifest fields differ from schema version 1",
    )
    _require(
        payload.get("execution_id") == execution_id,
        "source-panel manifest is not the next registered execution",
    )
    _require(
        payload.get("session_id") == session_id,
        "source-panel manifest names the wrong session",
    )
    artifacts_before = _artifact_snapshot(
        root,
        payload.get("artifacts"),
        verify_declared=True,
    )
    _validate_source_execution_manifest(
        root,
        source_relative,
        protocol=protocol,
        v4=v4,
        execution_id=execution_id,
        session_id=session_id,
        verify_file_hashes=True,
    )
    source_digest, source_bytes = _sha256_file(source)
    _require(
        (source_digest, source_bytes) == (source_digest_before, source_bytes_before),
        "source-panel staging manifest changed during validation",
    )
    artifacts_after = _artifact_snapshot(
        root,
        payload.get("artifacts"),
        verify_declared=False,
    )
    _require(
        artifacts_after == artifacts_before,
        "source-panel artifacts changed during validation",
    )

    final_relative = SOURCE_PANEL_MANIFEST_PATH.format(execution_id=execution_id)
    final_path = root / final_relative
    _assert_ordinary_file_or_missing(final_path, name="source-panel manifest")
    _require(not final_path.exists(), "source-panel manifest already exists")

    status_after = build_source_panel_status(
        repository_root,
        root,
        verify_file_hashes=True,
    )
    _require(
        status_after["status_sha256"] == status_before["status_sha256"],
        "source-panel status changed during staged preflight",
    )
    _require(
        not final_path.exists(),
        "staged preflight unexpectedly created the final source manifest",
    )

    repository = str(Path(repository_root).resolve())
    publication_command = [
        "causal4d",
        "protocol",
        "readiness",
        "source-panel-publish",
        repository,
        str(root),
        str(source),
    ]
    result: dict[str, Any] = {
        "schema_version": SOURCE_PANEL_STAGING_PREFLIGHT_SCHEMA_VERSION,
        "artifact_kind": SOURCE_PANEL_STAGING_PREFLIGHT_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "repository_root": repository,
        "dataset_root": str(root),
        "source_json": str(source),
        "source_manifest_relative_path": source_relative,
        "source_manifest_sha256": source_digest,
        "source_manifest_bytes": source_bytes,
        "execution_id": execution_id,
        "session_id": session_id,
        "source_panel_execution_index": next_execution["source_panel_execution_index"],
        "command_profile_id": next_execution["command_profile_id"],
        "artifact_count": len(artifacts_after),
        "artifacts": artifacts_after,
        "final_manifest_path": final_relative,
        "final_manifest_present": False,
        "specified_executions": status_before["specified_executions"],
        "validated_executions_before": status_before["validated_executions"],
        "validated_executions_after_publication": (
            status_before["validated_executions"] + 1
        ),
        "source_panel_evidence_sha256_before": status_before["evidence_sha256"],
        "source_panel_status_sha256_before": status_before["status_sha256"],
        "source_panel_status_stable": True,
        "publication_command_argv": publication_command,
        "publication_command_text": shlex.join(publication_command),
        "safe_to_publish": True,
        "published": False,
        "claim_bearing_evidence_mutated": False,
        "changes_registered_method": False,
        "valid": True,
        "complete": True,
        "passed": True,
        "target_outcomes_used": False,
    }
    result["evidence_sha256"] = source_panel_staging_evidence_sha256(result)
    result["status_sha256"] = source_panel_staging_status_sha256(result)
    return result


def write_source_panel_staging_preflight(
    path: str | Path,
    result: Mapping[str, Any],
) -> Path:
    """Atomically replace one derived staging-preflight report."""

    output = Path(path)
    atomic_write_json(output, dict(result))
    return output


__all__ = [
    "SOURCE_PANEL_STAGING_PREFLIGHT_ARTIFACT_KIND",
    "SOURCE_PANEL_STAGING_PREFLIGHT_SCHEMA_VERSION",
    "source_panel_staging_evidence_sha256",
    "source_panel_staging_status_sha256",
    "verify_source_panel_manifest_staging",
    "write_source_panel_staging_preflight",
]
