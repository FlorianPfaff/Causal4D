"""Read-only verification for a staged physical source-panel manifest."""

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
    _sha256_file,
    load_registered_preacquisition_chain,
    source_panel_execution_manifest_template,
)
from causal4d.preacquisition_source_panel_control import build_source_panel_status
from causal4d.preacquisition_source_validation import (
    _validate_source_execution_manifest,
)

STAGED_SOURCE_VERIFICATION_SCHEMA_VERSION = 1
STAGED_SOURCE_VERIFICATION_ARTIFACT_KIND = (
    "Causal4DStagedSourcePanelManifestVerification"
)


def staged_source_verification_evidence_sha256(
    values: Mapping[str, Any],
) -> str:
    """Return a mount-independent digest of one staged verification."""

    payload = deepcopy(dict(values))
    for field in (
        "repository_root",
        "dataset_root",
        "source_panel_status_sha256",
        "evidence_sha256",
        "status_sha256",
    ):
        payload.pop(field, None)
    return _canonical_sha256(payload, omitted_field="evidence_sha256")


def staged_source_verification_status_sha256(
    values: Mapping[str, Any],
) -> str:
    """Return the digest of the exact host-local verification report."""

    return _canonical_sha256(values, omitted_field="status_sha256")


def _resolved_dataset_root(dataset_root: str | Path) -> Path:
    candidate = Path(dataset_root)
    _assert_no_symlink_components(candidate, name="dataset root")
    _require(candidate.is_dir(), "dataset root must exist")
    return candidate.resolve()


def _staged_source_path(
    dataset_root: Path,
    source_json: str | Path,
    *,
    execution_id: str,
) -> tuple[Path, str]:
    source = Path(source_json)
    _assert_no_symlink_components(
        source,
        name="source-panel staged manifest",
    )
    _require(source.is_file(), "source-panel staged manifest is missing")
    resolved = source.resolve(strict=True)
    _require(
        resolved.is_relative_to(dataset_root),
        "source-panel staged manifest escapes the dataset root",
    )
    relative = resolved.relative_to(dataset_root)
    _require(
        relative.parent == Path("staging"),
        "source-panel staged manifest must be directly below dataset_root/staging",
    )
    _require(
        relative.name == f"{execution_id}.json",
        "source-panel staged manifest filename must match the next execution",
    )
    return resolved, relative.as_posix()


def _publication_command(source_relative: str) -> dict[str, Any]:
    argv = [
        "causal4d",
        "protocol",
        "readiness",
        "source-panel-publish",
        "${REPOSITORY_ROOT}",
        "${DATASET_ROOT}",
        f"${{DATASET_ROOT}}/{source_relative}",
    ]
    return {
        "argv_template": argv,
        "shell_template": shlex.join(argv),
    }


def verify_staged_source_panel_manifest(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
) -> dict[str, Any]:
    """Hash-verify the next staged source manifest without publishing it."""

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
    _require(
        isinstance(next_execution, Mapping),
        "source panel has no next execution",
    )
    _require(
        next_execution.get("template_present") is True
        and next_execution.get("template_valid") is True,
        "next source-panel manifest template is missing or invalid",
    )

    execution_id = str(next_execution["execution_id"])
    session_id = str(next_execution["session_id"])
    source, source_relative = _staged_source_path(
        root,
        source_json,
        execution_id=execution_id,
    )
    payload = _read_json_mapping(
        source,
        name="source-panel staged manifest",
    )
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

    final_relative = SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=execution_id
    )
    final_path = root / final_relative
    _assert_ordinary_file_or_missing(
        final_path,
        name="source-panel manifest",
    )
    _require(
        not final_path.exists(),
        "source-panel manifest already exists",
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
    source_sha256, source_bytes = _sha256_file(source)

    status_after = build_source_panel_status(
        repository_root,
        root,
        verify_file_hashes=True,
    )
    _require(
        status_after["status_sha256"] == status_before["status_sha256"],
        "source-panel status changed during staged verification",
    )
    _require(
        not final_path.exists(),
        "staged verification unexpectedly created the final manifest",
    )

    report: dict[str, Any] = {
        "schema_version": STAGED_SOURCE_VERIFICATION_SCHEMA_VERSION,
        "artifact_kind": STAGED_SOURCE_VERIFICATION_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "repository_root": str(Path(repository_root).resolve()),
        "dataset_root": str(root),
        "execution_id": execution_id,
        "session_id": session_id,
        "staged_manifest": {
            "path": source_relative,
            "sha256": source_sha256,
            "bytes": source_bytes,
        },
        "prospective_final_manifest_path": final_relative,
        "source_panel_evidence_sha256": status_before["evidence_sha256"],
        "source_panel_status_sha256": status_before["status_sha256"],
        "publication_command": _publication_command(source_relative),
        "mutated_dataset": False,
        "final_manifest_present": False,
        "changes_registered_method": False,
        "target_outcomes_used": False,
        "valid": True,
        "passed": True,
    }
    report["evidence_sha256"] = staged_source_verification_evidence_sha256(
        report
    )
    report["status_sha256"] = staged_source_verification_status_sha256(
        report
    )
    return report


def write_staged_source_panel_verification(
    path: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Atomically write one staged source-panel verification report."""

    output = Path(path)
    atomic_write_json(output, dict(report))
    return output


__all__ = [
    "STAGED_SOURCE_VERIFICATION_ARTIFACT_KIND",
    "STAGED_SOURCE_VERIFICATION_SCHEMA_VERSION",
    "staged_source_verification_evidence_sha256",
    "staged_source_verification_status_sha256",
    "verify_staged_source_panel_manifest",
    "write_staged_source_panel_verification",
]
