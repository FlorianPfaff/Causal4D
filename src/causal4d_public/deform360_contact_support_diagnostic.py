"""Source-only Deform360 diagnostic for support and contact realization."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json

from .deform360_contact_support_contract import (
    CONTACT_SUPPORT_CANDIDATE_POLICIES,
    CONTACT_SUPPORT_CONFIG_KIND,
    CONTACT_SUPPORT_DIAGNOSTIC_KIND,
    CONTACT_SUPPORT_DIAGNOSTIC_SCHEMA_VERSION,
    CONTACT_SUPPORT_NEGATIVE_CONTROL,
    CONTACT_SUPPORT_POLICIES,
    PREFIX_KINEMATICS_SUMMARY,
    SOURCE_FAILURE_SUMMARY,
    ContactSupportDiagnosticConfig,
    build_contact_support_decision,
    contact_support_config_sha256,
    contact_support_result_sha256,
    load_contact_support_diagnostic_lock,
    sha256_file,
    summarize_contact_support_policy,
    validate_source_contact_support_diagnostic,
    _require,
)
from .deform360_contact_support_episode import (
    build_contact_support_episode_record,
)
from .deform360_prefix_kinematics_diagnostic import (
    SOURCE_MILESTONE,
    verify_source_milestone,
)
from .deform360_replication import load_deform360_replication_protocol


def _verify_prior_diagnostic_locks(
    root: Path,
    lock_payload: Mapping[str, Any],
) -> dict[str, Any]:
    failure_path = root / SOURCE_FAILURE_SUMMARY
    prefix_path = root / PREFIX_KINEMATICS_SUMMARY
    _require(failure_path.is_file(), "source-failure attribution summary is missing")
    _require(prefix_path.is_file(), "prefix-kinematics summary is missing")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    _require(
        lock_payload["source_failure_attribution_result_sha256"]
        == failure["full_attribution_result_sha256"],
        "source-failure attribution identity changed",
    )
    _require(
        lock_payload["prefix_kinematics_result_sha256"]
        == prefix["artifacts"]["result_sha256"],
        "prefix-kinematics result identity changed",
    )
    _require(
        failure["decision"]["target_prefix_access_permitted"] is False
        and failure["decision"]["target_future_access_permitted"] is False,
        "source-failure attribution no longer preserves target closure",
    )
    _require(
        prefix["decision"]["passed"] is False
        and prefix["decision"]["target_prefix_access_permitted"] is False
        and prefix["decision"]["target_future_access_permitted"] is False,
        "prefix-kinematics result no longer records the negative target-closed result",
    )
    return {
        "source_failure_attribution": {
            "path": str(SOURCE_FAILURE_SUMMARY),
            "file_sha256": sha256_file(failure_path),
            "result_sha256": failure["full_attribution_result_sha256"],
        },
        "prefix_kinematics": {
            "path": str(PREFIX_KINEMATICS_SUMMARY),
            "file_sha256": sha256_file(prefix_path),
            "result_sha256": prefix["artifacts"]["result_sha256"],
        },
    }


def run_source_contact_support_diagnostic(
    repository_root: str | Path,
    protocol_path: str | Path,
    data_root: str | Path,
    official_phystwin_repo: str | Path,
    output_path: str | Path,
    *,
    lock_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run the locked support/contact mechanism comparison on source data."""

    lock = load_contact_support_diagnostic_lock(lock_path)
    config: ContactSupportDiagnosticConfig = lock["config"]
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    data = Path(data_root).resolve()
    official = Path(official_phystwin_repo).resolve()
    _require(root.is_dir(), "repository root is missing")
    _require(protocol_file.is_file(), "replication protocol is missing")
    _require(data.is_dir(), "Deform360 derived-data root is missing")
    _require(official.is_dir(), "official PhysTwin repository is missing")
    milestone_verification = verify_source_milestone(root)
    protocol = load_deform360_replication_protocol(protocol_file)
    lock_payload = lock["payload"]
    _require(
        lock_payload["protocol_config_sha256"] == protocol["config_sha256"],
        "locked replication protocol identity changed",
    )
    _require(
        lock_payload["source_milestone_manifest_sha256"]
        == milestone_verification["manifest_sha256"],
        "locked source milestone identity changed",
    )
    source_decision_path = (
        root / SOURCE_MILESTONE / "artifacts" / "source_backend_decision.json"
    )
    source_decision = json.loads(source_decision_path.read_text(encoding="utf-8"))
    _require(
        lock_payload["source_backend_decision_result_sha256"]
        == source_decision["result_sha256"],
        "locked source-backend decision identity changed",
    )
    prior_diagnostics = _verify_prior_diagnostic_locks(root, lock_payload)
    cohorts = {
        str(record["object_id"]): record for record in protocol["config"]["cohort"]
    }
    grid_root = root / SOURCE_MILESTONE / "artifacts" / "source-grids"
    _require(grid_root.is_dir(), "source-grid milestone directory is missing")
    selected_objects = tuple(lock["selected_object_ids"])
    _require(
        len(selected_objects) == len(set(selected_objects))
        and all(object_id in cohorts for object_id in selected_objects),
        "diagnostic object set is invalid",
    )
    records = []
    for object_id in selected_objects:
        paths = sorted((grid_root / object_id).glob("source_episode_*_grid.json"))
        expected = len(cohorts[object_id]["source_episode_ids"])
        _require(
            len(paths) == expected,
            f"{object_id} source-grid set is incomplete",
        )
        for path in paths:
            records.append(
                build_contact_support_episode_record(
                    repository_root=root,
                    data_root=data,
                    source_grid_path=path,
                    cohort=cohorts[object_id],
                    official_phystwin_repo=official,
                    device=device,
                    config=config,
                )
            )
    decision = build_contact_support_decision(records, config=config)
    payload = {
        "schema_version": CONTACT_SUPPORT_DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_kind": CONTACT_SUPPORT_DIAGNOSTIC_KIND,
        "config": config.as_dict(),
        "protocol": {
            "path": (
                str(protocol_file.relative_to(root))
                if protocol_file.is_relative_to(root)
                else str(protocol_file)
            ),
            "sha256": sha256_file(protocol_file),
            "config_sha256": protocol["config_sha256"],
        },
        "source_milestone": milestone_verification,
        "prior_diagnostics": prior_diagnostics,
        "diagnostic_lock": {
            "path": (
                str(lock["path"].relative_to(root))
                if lock["path"].is_relative_to(root)
                else str(lock["path"])
            ),
            "file_sha256": sha256_file(lock["path"]),
            "config_sha256": lock_payload["config_sha256"],
        },
        "selected_object_ids": list(selected_objects),
        "objects_without_complete_source_grids": sorted(
            set(cohorts) - set(selected_objects)
        ),
        "episode_records": records,
        "decision": decision,
        "information_boundary": {
            "source_candidate_outcomes_read": True,
            "source_future_geometry_read_for_scoring": True,
            "source_tactile_read": True,
            "source_robot_openings_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
            "registered_replication_result_changed": False,
            "registered_36_execution_method_changed": False,
        },
    }
    payload["result_sha256"] = contact_support_result_sha256(payload)
    atomic_write_json(output_path, payload)
    return payload


__all__ = [
    "CONTACT_SUPPORT_CANDIDATE_POLICIES",
    "CONTACT_SUPPORT_CONFIG_KIND",
    "CONTACT_SUPPORT_DIAGNOSTIC_KIND",
    "CONTACT_SUPPORT_DIAGNOSTIC_SCHEMA_VERSION",
    "CONTACT_SUPPORT_NEGATIVE_CONTROL",
    "CONTACT_SUPPORT_POLICIES",
    "ContactSupportDiagnosticConfig",
    "build_contact_support_decision",
    "contact_support_config_sha256",
    "contact_support_result_sha256",
    "load_contact_support_diagnostic_lock",
    "run_source_contact_support_diagnostic",
    "summarize_contact_support_policy",
    "validate_source_contact_support_diagnostic",
]
