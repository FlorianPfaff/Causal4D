from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_contact_support_diagnostic import (
    CONTACT_SUPPORT_CANDIDATE_POLICIES,
    CONTACT_SUPPORT_NEGATIVE_CONTROL,
    CONTACT_SUPPORT_POLICIES,
    ContactSupportDiagnosticConfig,
    build_contact_support_decision,
    contact_support_config_sha256,
    contact_support_result_sha256,
    load_contact_support_diagnostic_lock,
    summarize_contact_support_policy,
    validate_source_contact_support_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = (
    ROOT
    / "configs"
    / "causal4d_public"
    / "deform360_contact_support_v1.json"
)


def _policy(score: float, *, quality_valid: bool = True) -> dict[str, object]:
    return {
        "finite": True,
        "mean_chamfer_m": score,
        "late_chamfer_m": score,
        "p99_relative_edge_strain": 0.1,
        "maximum_relative_edge_strain": 0.2,
        "quality_valid": quality_valid,
    }


def _episode(
    index: int,
    *,
    candidate_scores: dict[str, float] | None = None,
    reproduction_passed: bool = True,
) -> dict[str, object]:
    overrides = candidate_scores or {}
    policies = {
        name: _policy(overrides.get(name, 1.0))
        for name in CONTACT_SUPPORT_POLICIES
    }
    return {
        "object_id": f"object-{index % 3}",
        "episode_id": f"object-{index % 3}/episode_{index:04d}",
        "policies": policies,
        "registered_baseline_reproduction": {"passed": reproduction_passed},
        "information_boundary": {
            "source_episode_only": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
            "method_selection_permitted": False,
        },
    }


def _valid_payload() -> dict[str, object]:
    records = [_episode(index) for index in range(30)]
    decision = build_contact_support_decision(
        records,
        config=ContactSupportDiagnosticConfig(),
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SourceContactSupportDiagnostic",
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
    return payload


def test_locked_contact_support_protocol_round_trips() -> None:
    lock = load_contact_support_diagnostic_lock(LOCK)

    assert lock["selected_object_ids"] == (
        "002-rope-silk",
        "081-stripe-rope",
        "083-blanket-cloth",
        "085-scarf-cloth",
        "170-spider",
    )
    assert lock["config"] == ContactSupportDiagnosticConfig()
    assert tuple(lock["payload"]["config"]["policy_order"]) == (
        CONTACT_SUPPORT_POLICIES
    )
    assert lock["payload"]["information_boundary"]["target_future_allowed"] is False


def test_lock_rejects_target_access(tmp_path: Path) -> None:
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    payload["information_boundary"]["target_future_allowed"] = True
    payload["config_sha256"] = contact_support_config_sha256(payload)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden information"):
        load_contact_support_diagnostic_lock(changed)


def test_policy_summary_retains_equal_episode_weighting_and_quality() -> None:
    records = [
        _episode(
            index,
            candidate_scores={"support_touching_v1": 0.8 if index < 20 else 1.2},
        )
        for index in range(30)
    ]
    records[0]["policies"]["support_touching_v1"] = _policy(
        0.7,
        quality_valid=False,
    )

    summary = summarize_contact_support_policy(records, "support_touching_v1")

    assert summary["common_finite_episode_count"] == 30
    assert summary["registered_mean_chamfer_m"] == pytest.approx(1.0)
    assert summary["quality_valid_episode_count"] == 29
    assert summary["registered_quality_valid_episode_count"] == 30
    assert summary["quality_regression_count"] == 1
    assert summary["win_fraction_vs_registered"] == pytest.approx(19 / 30)


def test_decision_can_support_a_contact_or_support_candidate_only() -> None:
    records = [
        _episode(
            index,
            candidate_scores={
                "support_touching_v1": 0.90,
                "contact_disabled_v1": 0.50,
            },
        )
        for index in range(30)
    ]

    decision = build_contact_support_decision(
        records,
        config=ContactSupportDiagnosticConfig(),
    )

    assert decision["baseline_reproduction_passed"] is True
    assert decision["supported_candidate_policies"] == ["support_touching_v1"]
    assert decision["any_candidate_supported"] is True
    assert CONTACT_SUPPORT_NEGATIVE_CONTROL not in CONTACT_SUPPORT_CANDIDATE_POLICIES
    assert decision["negative_control_is_not_promotable"] is True
    assert decision["target_future_access_permitted"] is False


def test_decision_fails_closed_when_baseline_does_not_reproduce() -> None:
    records = [
        _episode(
            index,
            candidate_scores={"support_touching_v1": 0.50},
            reproduction_passed=index != 0,
        )
        for index in range(30)
    ]

    decision = build_contact_support_decision(
        records,
        config=ContactSupportDiagnosticConfig(),
    )

    assert decision["baseline_reproduction_passed"] is False
    assert decision["supported_candidate_policies"] == []
    assert decision["any_candidate_supported"] is False


def test_result_validator_rejects_target_access() -> None:
    payload = _valid_payload()
    validate_source_contact_support_diagnostic(payload)
    changed = deepcopy(payload)
    changed["information_boundary"]["target_prefix_read"] = True
    changed["result_sha256"] = contact_support_result_sha256(changed)

    with pytest.raises(ValueError, match="information or claim boundary"):
        validate_source_contact_support_diagnostic(changed)


def test_result_validator_rejects_negative_control_promotion() -> None:
    payload = _valid_payload()
    changed = deepcopy(payload)
    changed["decision"]["supported_candidate_policies"] = [
        CONTACT_SUPPORT_NEGATIVE_CONTROL
    ]
    changed["result_sha256"] = contact_support_result_sha256(changed)

    with pytest.raises(ValueError, match="inadmissible candidate"):
        validate_source_contact_support_diagnostic(changed)
