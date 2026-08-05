"""Contracts and decision rules for the Deform360 contact/support diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np


CONTACT_SUPPORT_DIAGNOSTIC_SCHEMA_VERSION = 1
CONTACT_SUPPORT_DIAGNOSTIC_KIND = "Deform360SourceContactSupportDiagnostic"
CONTACT_SUPPORT_CONFIG_KIND = "Deform360SourceContactSupportConfig"
CONTACT_SUPPORT_POLICIES = (
    "registered_v1",
    "support_touching_v1",
    "support_lifted_5mm_v1",
    "contact_patch_4_v1",
    "contact_patch_12_v1",
    "opening_contact_schedule_v1",
    "contact_disabled_v1",
)
CONTACT_SUPPORT_CANDIDATE_POLICIES = CONTACT_SUPPORT_POLICIES[1:-1]
CONTACT_SUPPORT_NEGATIVE_CONTROL = CONTACT_SUPPORT_POLICIES[-1]
SOURCE_FAILURE_SUMMARY = Path(
    "milestones/deform360-source-failure-attribution-v1/summary.json"
)
PREFIX_KINEMATICS_SUMMARY = Path(
    "milestones/deform360-prefix-kinematics-v1/summary.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_mapping(value: Any, *, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(message)
    return cast(Mapping[str, Any], value)


def _require_nonempty_list(value: Any, *, message: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(message)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contact_support_result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _finite_float(value: Any, *, name: str) -> float:
    _require(
        type(value) in {int, float} and type(value) is not bool and np.isfinite(value),
        f"{name} must be a finite number",
    )
    return float(value)


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    _require(result >= 0.0, f"{name} must be nonnegative")
    return result


def _strict_positive_integer(value: Any, *, name: str) -> int:
    _require(type(value) is int and value >= 1, f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ContactSupportDiagnosticConfig:
    """Predeclared controls for the source-only contact/support comparison."""

    registered_taxel_count: int = 8
    narrow_taxel_count: int = 4
    wide_taxel_count: int = 12
    touching_clearance_m: float = 0.0
    lifted_clearance_m: float = 0.005
    baseline_chamfer_tolerance_m: float = 5e-4
    baseline_strain_tolerance: float = 1e-2
    minimum_relative_improvement: float = 0.05
    minimum_win_fraction: float = 0.60
    minimum_common_episode_count: int = 24

    def __post_init__(self) -> None:
        for name in (
            "registered_taxel_count",
            "narrow_taxel_count",
            "wide_taxel_count",
            "minimum_common_episode_count",
        ):
            object.__setattr__(
                self,
                name,
                _strict_positive_integer(getattr(self, name), name=name),
            )
        _require(
            self.narrow_taxel_count
            < self.registered_taxel_count
            < self.wide_taxel_count,
            "contact-patch sizes must bracket the registered patch",
        )
        for name in (
            "touching_clearance_m",
            "lifted_clearance_m",
            "baseline_chamfer_tolerance_m",
            "baseline_strain_tolerance",
            "minimum_relative_improvement",
            "minimum_win_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name=name),
            )
        _require(
            self.touching_clearance_m < 0.001 < self.lifted_clearance_m,
            "support policies must bracket the registered one-millimetre clearance",
        )
        _require(
            self.minimum_relative_improvement < 1.0,
            "minimum_relative_improvement must be below one",
        )
        _require(
            self.minimum_win_fraction <= 1.0,
            "minimum_win_fraction must be at most one",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def contact_support_config_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a diagnostic lock without its self-reported digest."""

    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _require_string_sequence(
    value: Any,
    *,
    message: str,
) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(type(item) is not str or not item for item in value)
    ):
        raise ValueError(message)
    return tuple(value)


def load_contact_support_diagnostic_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate the source-only contact/support diagnostic lock."""

    lock_path = Path(path).resolve()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == CONTACT_SUPPORT_DIAGNOSTIC_SCHEMA_VERSION,
        "contact/support config schema changed",
    )
    _require(
        payload.get("artifact_kind") == CONTACT_SUPPORT_CONFIG_KIND,
        "contact/support config kind changed",
    )
    _require(
        payload.get("config_sha256") == contact_support_config_sha256(payload),
        "contact/support config checksum mismatch",
    )
    controls = _require_mapping(
        payload.get("config"),
        message="contact/support controls are missing",
    )
    selected = _require_nonempty_list(
        controls.get("selected_object_ids"),
        message="selected source object set is invalid",
    )
    _require(
        all(type(value) is str and value for value in selected)
        and len(selected) == len(set(selected)),
        "selected source object set is invalid",
    )
    _require(
        controls.get("candidate_selection")
        == "quality_constrained_source_oracle_else_finite_oracle",
        "source candidate-selection policy changed",
    )
    _require(
        _require_string_sequence(
            controls.get("policy_order"),
            message="contact/support policy order is invalid",
        )
        == CONTACT_SUPPORT_POLICIES,
        "contact/support policy set or ordering changed",
    )
    _require(
        _require_string_sequence(
            controls.get("candidate_policies"),
            message="contact/support candidate set is invalid",
        )
        == CONTACT_SUPPORT_CANDIDATE_POLICIES,
        "contact/support candidate set changed",
    )
    _require(
        controls.get("negative_control_policy") == CONTACT_SUPPORT_NEGATIVE_CONTROL,
        "contact/support negative control changed",
    )
    boundary = _require_mapping(
        payload.get("information_boundary"),
        message="contact/support information boundary is missing",
    )
    _require(
        boundary.get("source_only") is True
        and boundary.get("calibration_outcomes_allowed") is False
        and boundary.get("target_prefix_allowed") is False
        and boundary.get("target_future_allowed") is False
        and boundary.get("registered_method_mutable") is False,
        "contact/support lock opens a forbidden information or method boundary",
    )
    config = ContactSupportDiagnosticConfig(
        registered_taxel_count=controls["registered_taxel_count"],
        narrow_taxel_count=controls["narrow_taxel_count"],
        wide_taxel_count=controls["wide_taxel_count"],
        touching_clearance_m=controls["touching_clearance_m"],
        lifted_clearance_m=controls["lifted_clearance_m"],
        baseline_chamfer_tolerance_m=controls["baseline_chamfer_tolerance_m"],
        baseline_strain_tolerance=controls["baseline_strain_tolerance"],
        minimum_relative_improvement=controls["minimum_relative_improvement"],
        minimum_win_fraction=controls["minimum_win_fraction"],
        minimum_common_episode_count=controls["minimum_common_episode_count"],
    )
    return {
        "path": lock_path,
        "payload": payload,
        "config": config,
        "selected_object_ids": tuple(selected),
    }


def summarize_contact_support_policy(
    episode_records: Sequence[Mapping[str, Any]],
    policy_name: str,
) -> dict[str, Any]:
    """Aggregate one policy with equal episode weight against the baseline."""

    _require(policy_name in CONTACT_SUPPORT_POLICIES, "unknown contact/support policy")
    scores: list[float] = []
    baseline_scores: list[float] = []
    wins: list[bool] = []
    valid_count = 0
    baseline_valid_count = 0
    rescues = 0
    regressions = 0
    per_object: dict[str, list[tuple[float, float, bool, bool]]] = {}
    for record in episode_records:
        policies = _require_mapping(
            record.get("policies"),
            message="episode policy record is missing",
        )
        baseline = _require_mapping(
            policies.get("registered_v1"),
            message="episode registered result is missing",
        )
        candidate = _require_mapping(
            policies.get(policy_name),
            message="episode candidate result is missing",
        )
        if baseline.get("finite") is not True or candidate.get("finite") is not True:
            continue
        baseline_score = _finite_float(
            baseline.get("mean_chamfer_m"),
            name="registered mean_chamfer_m",
        )
        candidate_score = _finite_float(
            candidate.get("mean_chamfer_m"),
            name="candidate mean_chamfer_m",
        )
        _require(
            baseline_score > 0.0 and candidate_score > 0.0,
            "finite Chamfer scores must be positive",
        )
        baseline_valid = baseline.get("quality_valid") is True
        candidate_valid = candidate.get("quality_valid") is True
        candidate_win = bool(candidate_valid and candidate_score < baseline_score)
        scores.append(candidate_score)
        baseline_scores.append(baseline_score)
        wins.append(candidate_win)
        valid_count += int(candidate_valid)
        baseline_valid_count += int(baseline_valid)
        rescues += int(not baseline_valid and candidate_valid)
        regressions += int(baseline_valid and not candidate_valid)
        per_object.setdefault(str(record["object_id"]), []).append(
            (candidate_score, baseline_score, candidate_valid, candidate_win)
        )
    if not scores:
        return {
            "policy": policy_name,
            "common_finite_episode_count": 0,
            "mean_chamfer_m": None,
            "registered_mean_chamfer_m": None,
            "relative_improvement_vs_registered": None,
            "win_fraction_vs_registered": None,
            "quality_valid_episode_count": 0,
            "registered_quality_valid_episode_count": 0,
            "quality_rescue_count": 0,
            "quality_regression_count": 0,
            "per_object": {},
        }
    mean_score = float(np.mean(scores))
    mean_baseline = float(np.mean(baseline_scores))
    object_summary: dict[str, Any] = {}
    for object_id, rows in sorted(per_object.items()):
        object_scores = np.asarray([row[0] for row in rows], dtype=float)
        object_baseline = np.asarray([row[1] for row in rows], dtype=float)
        object_summary[object_id] = {
            "episode_count": len(rows),
            "mean_chamfer_m": float(np.mean(object_scores)),
            "registered_mean_chamfer_m": float(np.mean(object_baseline)),
            "relative_improvement_vs_registered": float(
                (np.mean(object_baseline) - np.mean(object_scores))
                / np.mean(object_baseline)
            ),
            "quality_valid_episode_count": int(sum(row[2] for row in rows)),
            "win_fraction_vs_registered": float(np.mean([row[3] for row in rows])),
        }
    return {
        "policy": policy_name,
        "common_finite_episode_count": len(scores),
        "mean_chamfer_m": mean_score,
        "registered_mean_chamfer_m": mean_baseline,
        "relative_improvement_vs_registered": (mean_baseline - mean_score)
        / mean_baseline,
        "win_fraction_vs_registered": float(np.mean(wins)),
        "quality_valid_episode_count": valid_count,
        "registered_quality_valid_episode_count": baseline_valid_count,
        "quality_rescue_count": rescues,
        "quality_regression_count": regressions,
        "per_object": object_summary,
    }


def _candidate_passes(
    summary: Mapping[str, Any],
    *,
    config: ContactSupportDiagnosticConfig,
    reproduction_passed: bool,
) -> bool:
    improvement = summary.get("relative_improvement_vs_registered")
    win_fraction = summary.get("win_fraction_vs_registered")
    return bool(
        reproduction_passed
        and summary.get("common_finite_episode_count", 0)
        >= config.minimum_common_episode_count
        and improvement is not None
        and improvement >= config.minimum_relative_improvement
        and win_fraction is not None
        and win_fraction >= config.minimum_win_fraction
        and summary.get("quality_valid_episode_count", 0)
        >= summary.get("registered_quality_valid_episode_count", 0)
    )


def build_contact_support_decision(
    episode_records: Sequence[Mapping[str, Any]],
    *,
    config: ContactSupportDiagnosticConfig,
) -> dict[str, Any]:
    """Apply the source gate independently to each physical mechanism."""

    reproduction = [
        bool(record.get("registered_baseline_reproduction", {}).get("passed"))
        for record in episode_records
    ]
    reproduction_passed = bool(reproduction and all(reproduction))
    summaries = {
        policy: summarize_contact_support_policy(episode_records, policy)
        for policy in CONTACT_SUPPORT_POLICIES
    }
    supported = [
        policy
        for policy in CONTACT_SUPPORT_CANDIDATE_POLICIES
        if _candidate_passes(
            summaries[policy],
            config=config,
            reproduction_passed=reproduction_passed,
        )
    ]
    return {
        "registered_policy": "registered_v1",
        "candidate_policies": list(CONTACT_SUPPORT_CANDIDATE_POLICIES),
        "negative_control_policy": CONTACT_SUPPORT_NEGATIVE_CONTROL,
        "baseline_reproduction_passed": reproduction_passed,
        "baseline_reproduction_episode_count": len(reproduction),
        "minimum_common_episode_count": config.minimum_common_episode_count,
        "minimum_relative_improvement": config.minimum_relative_improvement,
        "minimum_win_fraction": config.minimum_win_fraction,
        "require_non_decreasing_quality_valid_count": True,
        "policy_summaries": summaries,
        "supported_candidate_policies": supported,
        "any_candidate_supported": bool(supported),
        "interpretation": (
            "one or more contact/support mechanisms pass the source-only gate"
            if supported
            else "no contact/support mechanism passes the source-only gate"
        ),
        "negative_control_is_not_promotable": True,
        "registered_method_changed": False,
        "target_prefix_access_permitted": False,
        "target_future_access_permitted": False,
    }


def validate_source_contact_support_diagnostic(payload: Mapping[str, Any]) -> None:
    """Validate a result and its target-closed claim boundary."""

    _require(
        payload.get("schema_version") == CONTACT_SUPPORT_DIAGNOSTIC_SCHEMA_VERSION,
        "contact/support diagnostic schema changed",
    )
    _require(
        payload.get("artifact_kind") == CONTACT_SUPPORT_DIAGNOSTIC_KIND,
        "contact/support diagnostic kind changed",
    )
    _require(
        payload.get("result_sha256") == contact_support_result_sha256(payload),
        "contact/support diagnostic checksum mismatch",
    )
    boundary = _require_mapping(
        payload.get("information_boundary"),
        message="contact/support diagnostic boundary is missing",
    )
    _require(
        boundary.get("calibration_outcomes_read") is False
        and boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False
        and boundary.get("registered_replication_result_changed") is False
        and boundary.get("registered_36_execution_method_changed") is False,
        "contact/support diagnostic crossed its information or claim boundary",
    )
    records = _require_nonempty_list(
        payload.get("episode_records"),
        message="contact/support diagnostic has no episodes",
    )
    for raw_record in records:
        record = _require_mapping(
            raw_record,
            message="contact/support episode is not a mapping",
        )
        episode_boundary = _require_mapping(
            record.get("information_boundary"),
            message="contact/support episode boundary is missing",
        )
        _require(
            episode_boundary.get("source_episode_only") is True
            and episode_boundary.get("calibration_outcomes_read") is False
            and episode_boundary.get("target_prefix_read") is False
            and episode_boundary.get("target_future_read") is False
            and episode_boundary.get("method_selection_permitted") is False,
            "contact/support episode crossed the source-only boundary",
        )
        policies = _require_mapping(
            record.get("policies"),
            message="contact/support episode policies are missing",
        )
        _require(
            tuple(policies) == CONTACT_SUPPORT_POLICIES,
            "contact/support policy set or ordering changed",
        )
    decision = _require_mapping(
        payload.get("decision"),
        message="contact/support decision is missing",
    )
    _require(
        decision.get("registered_method_changed") is False
        and decision.get("target_prefix_access_permitted") is False
        and decision.get("target_future_access_permitted") is False
        and decision.get("negative_control_is_not_promotable") is True,
        "contact/support decision crossed its method or target boundary",
    )
    supported = decision.get("supported_candidate_policies")
    _require(
        isinstance(supported, list)
        and all(policy in CONTACT_SUPPORT_CANDIDATE_POLICIES for policy in supported),
        "contact/support decision names an inadmissible candidate",
    )


__all__ = [
    "CONTACT_SUPPORT_CANDIDATE_POLICIES",
    "CONTACT_SUPPORT_CONFIG_KIND",
    "CONTACT_SUPPORT_DIAGNOSTIC_KIND",
    "CONTACT_SUPPORT_DIAGNOSTIC_SCHEMA_VERSION",
    "CONTACT_SUPPORT_NEGATIVE_CONTROL",
    "CONTACT_SUPPORT_POLICIES",
    "PREFIX_KINEMATICS_SUMMARY",
    "SOURCE_FAILURE_SUMMARY",
    "ContactSupportDiagnosticConfig",
    "build_contact_support_decision",
    "contact_support_config_sha256",
    "contact_support_result_sha256",
    "load_contact_support_diagnostic_lock",
    "sha256_file",
    "summarize_contact_support_policy",
    "validate_source_contact_support_diagnostic",
]
