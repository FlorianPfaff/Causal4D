"""Seal and verify the Causal4D confirmatory real-experiment method."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
MILESTONE_ID = "causal4d-same-object-real-v1"
BPT_PIN_PATH = "requirements/ci/bayesian-phystwin-provider-v1.sha"
PROTOCOL_PATH = "configs/causal4d/sloth_multi_action_v1.json"
PREACQUISITION_PATH = "configs/causal4d/sloth_preacquisition_v4.json"
PREACQUISITION_PLAN_ID = "causal4d-sloth-preacquisition-v4"
MECHANISM_GATE_EVIDENCE_PATH = (
    "runs/causal4d_preacquisition_v4/mechanism_gate_controls.json"
)
REQUIRED_LOCKED_PATHS = (
    PROTOCOL_PATH,
    "configs/causal4d/sloth_multi_action_v1_schedule.csv",
    PREACQUISITION_PATH,
    MECHANISM_GATE_EVIDENCE_PATH,
    "docs/causal4d_paper_scope.md",
    "docs/causal4d_same_object_multi_action_protocol.md",
    "docs/causal4d_real_experiment_milestone.md",
    "docs/causal4d_preacquisition_v4.md",
    "docs/execution_block_conformal_calibration.md",
    "src/causal4d/execution_block_calibration.py",
    "src/causal4d/cli/execution_block_calibration.py",
    BPT_PIN_PATH,
    "pyproject.toml",
)
REQUIRED_ANALYSIS_ENTRYPOINTS = (
    "causal4d-real-protocol",
    "causal4d-execution-block-calibration",
    "causal4d-evaluate-physical-counterfactual",
)
DIAGNOSTIC_ONLY_ANALYSIS_ENTRYPOINTS = ("causal4d-real-calibration",)
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_relative_path(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "locked file path is missing")
    path = Path(value)
    _require(
        not path.is_absolute() and ".." not in path.parts,
        "locked file path is unsafe",
    )
    return path.as_posix()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _canonical_payload_sha256(
    values: Mapping[str, Any],
    *,
    omitted_field: str,
) -> str:
    payload = dict(values)
    payload.pop(omitted_field, None)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), f"{name} must be a JSON object")
    return dict(payload)


def _read_bayesian_phystwin_pin(pin_path: Path) -> str:
    value = pin_path.read_text(encoding="utf-8").strip()
    _require(
        bool(_SHA40.fullmatch(value)),
        "Bayesian-PhysTwin pin must contain one lowercase 40-hex commit",
    )
    return value


def _validate_utc_timestamp(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "freeze timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("freeze timestamp is not ISO 8601") from error
    _require(parsed.tzinfo is not None, "freeze timestamp must include a timezone")
    _require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        "freeze timestamp must be UTC",
    )
    return value


def _preacquisition_contract(
    repository_root: Path,
    *,
    protocol_design_sha256: str,
) -> dict[str, Any]:
    plan = _read_json_object(
        repository_root / PREACQUISITION_PATH,
        name="pre-acquisition amendment",
    )
    _require(
        plan.get("plan_id") == PREACQUISITION_PLAN_ID,
        "wrong pre-acquisition plan id",
    )
    _require(
        plan.get("status") == "supersedes_v3_before_any_physical_execution",
        "pre-acquisition v4 was not locked before physical execution",
    )
    amendment_sha256 = plan.get("amendment_sha256")
    _require(
        isinstance(amendment_sha256, str) and bool(_SHA64.fullmatch(amendment_sha256)),
        "pre-acquisition amendment SHA-256 is missing",
    )
    _require(
        amendment_sha256
        == _canonical_payload_sha256(plan, omitted_field="amendment_sha256"),
        "pre-acquisition amendment SHA-256 does not match its contents",
    )
    _require(
        plan.get("supersedes", {}).get(
            "physical_executions_completed_before_supersession"
        )
        == 0,
        "pre-acquisition v4 was introduced after physical execution began",
    )

    base_protocol = plan.get("base_protocol", {})
    _require(
        base_protocol.get("design_sha256") == protocol_design_sha256,
        "pre-acquisition amendment binds a different base protocol",
    )
    _require(
        base_protocol.get("confirmatory_execution_count") == 36,
        "pre-acquisition amendment changed the confirmatory execution count",
    )

    gate_lock = plan.get("mechanism_gate_control_lock", {})
    _require(
        gate_lock.get("evidence_artifact") == MECHANISM_GATE_EVIDENCE_PATH,
        "pre-acquisition amendment references the wrong gate-control evidence",
    )
    evidence = _read_json_object(
        repository_root / MECHANISM_GATE_EVIDENCE_PATH,
        name="mechanism-gate control evidence",
    )
    _require(
        evidence.get("schema_version") == 1
        and evidence.get("artifact_kind") == "MechanismGateControlEvidence",
        "unexpected mechanism-gate control evidence",
    )
    evidence_sha256 = evidence.get("result_sha256")
    _require(
        isinstance(evidence_sha256, str) and bool(_SHA64.fullmatch(evidence_sha256)),
        "mechanism-gate control result SHA-256 is missing",
    )
    _require(
        evidence_sha256
        == _canonical_payload_sha256(evidence, omitted_field="result_sha256"),
        "mechanism-gate control checksum mismatch",
    )
    _require(
        gate_lock.get("evidence_sha256") == evidence_sha256,
        "pre-acquisition amendment binds different gate-control evidence",
    )
    checks = evidence.get("acceptance_checks", {})
    _require(
        checks
        == {
            "placebo_null_full_gate_upper_below_5_percent": True,
            "positive_control_full_gate_lower_above_80_percent": True,
            "wrong_family_on_positive_upper_below_5_percent": True,
        }
        and evidence.get("frozen_v3_gate_supported_in_controlled_benchmark") is True,
        "mechanism-gate controls did not pass the frozen acceptance checks",
    )

    return {
        "path": PREACQUISITION_PATH,
        "plan_id": PREACQUISITION_PLAN_ID,
        "amendment_sha256": amendment_sha256,
        "base_protocol_design_sha256": protocol_design_sha256,
        "confirmatory_execution_count": 36,
        "mechanism_gate_control": {
            "path": MECHANISM_GATE_EVIDENCE_PATH,
            "result_sha256": evidence_sha256,
        },
    }


def _analysis_contract() -> dict[str, Any]:
    return {
        "entrypoints": list(REQUIRED_ANALYSIS_ENTRYPOINTS),
        "diagnostic_only_entrypoints": list(DIAGNOSTIC_ONLY_ANALYSIS_ENTRYPOINTS),
        "allowed_observation_prefix_frames": 6,
        "confirmatory_calibration": {
            "entrypoint": "causal4d-execution-block-calibration",
            "confidence_level": 0.90,
            "outer_fold_count": 12,
            "expected_calibration_units_per_outer_fold": 9,
            "order_statistic_rank_one_based": 9,
            "calibration_unit": (
                "one preregistered execution per independent session"
            ),
            "score_kind": "max_abs_standardized_coordinate_v1",
            "target_threshold_reselection_allowed": False,
            "pooled_coordinate_conformal_claimed": False,
            "worst_group_coverage_guarantee_claimed": False,
        },
        "target_outcomes_may_select_method_or_hyperparameters": False,
        "optional_branches_may_change_primary_analysis": False,
        "method_changes_require_new_protocol_version": True,
    }


def _reporting_contract() -> dict[str, Any]:
    return {
        "report_success_or_well_powered_negative_result": True,
        "report_all_36_executions_or_preregistered_exclusions": True,
        "report_independent_execution_calibration": True,
        "report_effect_intervals_and_replay_reset_variance": True,
        "optional_semantic_or_public_data_results_cannot_rescue_primary_failure": True,
    }


def repository_git_state(repository_root: str | Path) -> dict[str, Any]:
    """Return the commit and cleanliness of the checkout used for acquisition."""

    root = Path(repository_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_lines = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("repository root is not a readable Git checkout") from error
    _require(bool(_SHA40.fullmatch(commit)), "Git HEAD is not a full commit SHA")
    return {"commit_sha": commit, "dirty_worktree": bool(dirty_lines)}


def validate_repository_checkout(
    manifest: Mapping[str, Any], repository_root: str | Path
) -> dict[str, Any]:
    """Require the deployed checkout to be clean and at the frozen commit."""

    state = repository_git_state(repository_root)
    _require(not state["dirty_worktree"], "acquisition checkout is dirty")
    frozen_commit = manifest.get("causal4d", {}).get("commit_sha")
    _require(
        state["commit_sha"] == frozen_commit,
        "checkout does not match frozen Causal4D commit",
    )
    return state


def build_method_freeze_manifest(
    repository_root: str | Path,
    *,
    causal4d_commit_sha: str,
    frozen_by: str,
    frozen_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a sealed manifest for the exact method used during acquisition."""

    root = Path(repository_root)
    _require(
        bool(_SHA40.fullmatch(causal4d_commit_sha)),
        "Causal4D commit must be a full SHA",
    )
    _require(
        isinstance(frozen_by, str) and bool(frozen_by.strip()),
        "frozen_by is required",
    )
    timestamp = _validate_utc_timestamp(
        frozen_at_utc or datetime.now(timezone.utc).isoformat()
    )

    locked_files = []
    for relative in REQUIRED_LOCKED_PATHS:
        path = root / relative
        _require(path.is_file(), f"required freeze file is missing: {relative}")
        digest, size = _sha256_file(path)
        locked_files.append({"path": relative, "sha256": digest, "bytes": size})

    protocol = _read_json_object(root / PROTOCOL_PATH, name="real protocol")
    design_sha256 = protocol.get("design_sha256")
    _require(
        isinstance(design_sha256, str) and bool(_SHA64.fullmatch(design_sha256)),
        "protocol design SHA-256 is missing",
    )
    preacquisition = _preacquisition_contract(
        root,
        protocol_design_sha256=design_sha256,
    )
    bpt_commit = _read_bayesian_phystwin_pin(root / BPT_PIN_PATH)

    return {
        "schema_version": SCHEMA_VERSION,
        "milestone_id": MILESTONE_ID,
        "status": "sealed",
        "frozen_at_utc": timestamp,
        "frozen_by": frozen_by.strip(),
        "locked_before_confirmatory_collection": True,
        "target_outcomes_observed_at_freeze": False,
        "causal4d": {
            "commit_sha": causal4d_commit_sha,
            "dirty_worktree": False,
        },
        "bayesian_phystwin": {"commit_sha": bpt_commit},
        "protocol": {
            "path": PROTOCOL_PATH,
            "design_sha256": design_sha256,
        },
        "preacquisition": preacquisition,
        "locked_files": locked_files,
        "analysis_contract": _analysis_contract(),
        "reporting_contract": _reporting_contract(),
    }


def write_method_freeze_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def load_method_freeze_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_method_freeze_manifest(
    manifest: Mapping[str, Any],
    repository_root: str | Path,
    *,
    expected_causal4d_commit_sha: str | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate preregistration timing, code pins, file hashes, and claim boundaries."""

    root = Path(repository_root)
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "unsupported freeze schema",
    )
    _require(
        manifest.get("milestone_id") == MILESTONE_ID,
        "wrong real-experiment milestone",
    )
    _require(manifest.get("status") == "sealed", "method freeze is not sealed")
    _require(
        manifest.get("locked_before_confirmatory_collection") is True,
        "method was not locked before confirmatory collection",
    )
    _require(
        manifest.get("target_outcomes_observed_at_freeze") is False,
        "target outcomes were observed before the method freeze",
    )
    _validate_utc_timestamp(manifest.get("frozen_at_utc"))
    _require(
        isinstance(manifest.get("frozen_by"), str)
        and bool(manifest["frozen_by"].strip()),
        "freeze signer is missing",
    )

    causal4d = manifest.get("causal4d", {})
    commit_sha = causal4d.get("commit_sha")
    _require(
        isinstance(commit_sha, str) and bool(_SHA40.fullmatch(commit_sha)),
        "invalid Causal4D commit SHA",
    )
    _require(causal4d.get("dirty_worktree") is False, "acquisition checkout was dirty")
    if expected_causal4d_commit_sha is not None:
        _require(
            commit_sha == expected_causal4d_commit_sha,
            "checkout does not match frozen Causal4D commit",
        )

    bpt_commit = manifest.get("bayesian_phystwin", {}).get("commit_sha")
    _require(
        bpt_commit == _read_bayesian_phystwin_pin(root / BPT_PIN_PATH),
        "Bayesian-PhysTwin pin differs from the frozen dependency",
    )

    protocol = manifest.get("protocol", {})
    _require(protocol.get("path") == PROTOCOL_PATH, "wrong protocol path")
    checked_protocol = _read_json_object(root / PROTOCOL_PATH, name="real protocol")
    _require(
        protocol.get("design_sha256") == checked_protocol.get("design_sha256"),
        "protocol design digest differs from the freeze",
    )
    checked_preacquisition = _preacquisition_contract(
        root,
        protocol_design_sha256=str(protocol["design_sha256"]),
    )
    _require(
        manifest.get("preacquisition") == checked_preacquisition,
        "pre-acquisition contract differs from the registered method freeze",
    )

    entries = manifest.get("locked_files", [])
    _require(isinstance(entries, list), "locked_files must be a list")
    by_path: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        _require(isinstance(entry, Mapping), "locked file descriptor is invalid")
        relative = _safe_relative_path(entry.get("path"))
        _require(relative not in by_path, f"duplicate locked file: {relative}")
        _require(
            isinstance(entry.get("sha256"), str)
            and bool(_SHA64.fullmatch(entry["sha256"])),
            f"locked file checksum is invalid: {relative}",
        )
        _require(
            isinstance(entry.get("bytes"), int)
            and not isinstance(entry["bytes"], bool)
            and entry["bytes"] >= 0,
            f"locked file byte count is invalid: {relative}",
        )
        by_path[relative] = entry
    _require(
        set(by_path) == set(REQUIRED_LOCKED_PATHS),
        "locked file set is incomplete or expanded",
    )
    if verify_files:
        for relative in REQUIRED_LOCKED_PATHS:
            path = root / relative
            _require(path.is_file(), f"locked file is missing: {relative}")
            digest, size = _sha256_file(path)
            descriptor = by_path[relative]
            _require(
                descriptor.get("sha256") == digest,
                f"locked file checksum mismatch: {relative}",
            )
            _require(
                descriptor.get("bytes") == size,
                f"locked file byte count mismatch: {relative}",
            )

    _require(
        manifest.get("analysis_contract") == _analysis_contract(),
        "analysis contract differs from the registered method freeze",
    )
    _require(
        manifest.get("reporting_contract") == _reporting_contract(),
        "reporting contract differs from the registered milestone",
    )
    return {
        "milestone_id": MILESTONE_ID,
        "causal4d_commit_sha": commit_sha,
        "bayesian_phystwin_commit_sha": bpt_commit,
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_amendment_sha256": checked_preacquisition[
            "amendment_sha256"
        ],
        "mechanism_gate_control_sha256": checked_preacquisition[
            "mechanism_gate_control"
        ]["result_sha256"],
        "confirmatory_calibration_entrypoint": _analysis_contract()[
            "confirmatory_calibration"
        ]["entrypoint"],
        "locked_files_checked": len(REQUIRED_LOCKED_PATHS),
        "file_hashes_verified": verify_files,
        "passed": True,
    }
