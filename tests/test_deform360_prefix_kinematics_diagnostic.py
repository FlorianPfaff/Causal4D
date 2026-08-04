from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_prefix_kinematics import (
    PREFIX_KINEMATICS_POLICIES,
)
from causal4d_public.deform360_prefix_kinematics_diagnostic import (
    PREFIX_KINEMATICS_DIAGNOSTIC_KIND,
    PREFIX_KINEMATICS_DIAGNOSTIC_SCHEMA_VERSION,
    PrefixKinematicsDiagnosticConfig,
    build_source_decision,
    load_prefix_kinematics_diagnostic_lock,
    select_fixed_source_candidate,
    summarize_policy,
    validate_source_prefix_kinematics_diagnostic,
    verify_source_milestone,
)


ROOT = Path(__file__).resolve().parents[1]
GRID_ROOT = (
    ROOT
    / "milestones"
    / "deform360-replication-source-backend-v1"
    / "artifacts"
    / "source-grids"
)


def _load_grid(object_id: str, episode_id: int) -> dict:
    path = GRID_ROOT / object_id / f"source_episode_{episode_id:04d}_grid.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _policy(
    score: float,
    *,
    quality_valid: bool = True,
    finite: bool = True,
) -> dict:
    return {
        "finite": finite,
        "mean_chamfer_m": score if finite else None,
        "quality_valid": quality_valid,
    }


def _episode(
    object_id: str,
    zero: float,
    graph: float,
    *,
    zero_quality: bool = True,
    graph_quality: bool = True,
    reproduction: bool = True,
) -> dict:
    return {
        "object_id": object_id,
        "policies": {
            "zero_v1": _policy(zero, quality_valid=zero_quality),
            "global_contact_translation_v1": _policy(
                (zero + graph) / 2.0,
                quality_valid=graph_quality,
            ),
            "graph_harmonic_contact_v1": _policy(
                graph,
                quality_valid=graph_quality,
            ),
        },
        "zero_baseline_reproduction": {"passed": reproduction},
    }


def _canonical_sha256(payload: dict) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_locked_config_loads_exact_controls() -> None:
    lock = load_prefix_kinematics_diagnostic_lock(
        ROOT
        / "configs"
        / "causal4d_public"
        / "deform360_prefix_kinematics_v1.json"
    )
    assert lock["payload"]["config_sha256"] == (
        "18b9554b80bd6f4cd39813323267553df586403e032d711df0517c93b012bb27"
    )
    assert lock["selected_object_ids"] == (
        "002-rope-silk",
        "081-stripe-rope",
        "083-blanket-cloth",
        "085-scarf-cloth",
        "170-spider",
    )
    assert lock["config"].minimum_common_episode_count == 24


def test_locked_config_rejects_checksum_drift(tmp_path) -> None:
    source = (
        ROOT
        / "configs"
        / "causal4d_public"
        / "deform360_prefix_kinematics_v1.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["config"]["kinematics"]["lookback_frames"] = 4
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_prefix_kinematics_diagnostic_lock(changed)


def test_frozen_source_milestone_verifies() -> None:
    result = verify_source_milestone(ROOT)
    assert result["verified_file_count"] >= 50
    assert len(result["manifest_sha256"]) == 64


def test_selection_uses_quality_constrained_source_oracle() -> None:
    selected = select_fixed_source_candidate(_load_grid("081-stripe-rope", 1))
    assert selected["selection_kind"] == "quality_constrained_source_oracle"
    assert selected["archived_quality_valid"] is True
    assert selected["archived_p99_relative_edge_strain"] <= 0.5


def test_selection_retains_unconstrained_episode_when_no_candidate_is_valid() -> None:
    selected = select_fixed_source_candidate(_load_grid("083-blanket-cloth", 8))
    assert selected["selection_kind"] == "finite_unconstrained_source_oracle"
    assert selected["archived_quality_valid"] is False
    assert selected["archived_p99_relative_edge_strain"] > 0.5


def test_selection_rejects_changed_grid_checksum() -> None:
    grid = _load_grid("002-rope-silk", 0)
    grid["candidate_scores"][0]["mean_chamfer_m"] += 0.001
    with pytest.raises(ValueError, match="checksum"):
        select_fixed_source_candidate(grid)


def test_policy_summary_counts_rescues_and_regressions() -> None:
    records = [
        _episode("rope", 0.04, 0.03),
        _episode(
            "rope",
            0.05,
            0.04,
            zero_quality=False,
            graph_quality=True,
        ),
        _episode(
            "cloth",
            0.03,
            0.02,
            zero_quality=True,
            graph_quality=False,
        ),
    ]
    summary = summarize_policy(records, "graph_harmonic_contact_v1")
    assert summary["common_finite_episode_count"] == 3
    assert summary["quality_rescue_count"] == 1
    assert summary["quality_regression_count"] == 1
    assert summary["win_fraction_vs_zero"] == pytest.approx(2 / 3)
    assert summary["relative_improvement_vs_zero"] > 0.2
    assert set(summary["per_object"]) == {"cloth", "rope"}


def test_decision_passes_only_with_reproduced_non_degrading_gain() -> None:
    records = [
        _episode(f"object-{index % 2}", 0.04, 0.03)
        for index in range(6)
    ]
    config = PrefixKinematicsDiagnosticConfig(
        minimum_common_episode_count=6,
        minimum_relative_improvement=0.05,
        minimum_win_fraction=0.60,
    )
    decision = build_source_decision(records, config=config)
    assert decision["passed"] is True
    assert decision["baseline_reproduction_passed"] is True
    assert decision["target_prefix_access_permitted"] is False

    changed = copy.deepcopy(records)
    changed[0]["zero_baseline_reproduction"]["passed"] = False
    assert build_source_decision(changed, config=config)["passed"] is False


def test_decision_rejects_loss_of_quality_valid_episodes() -> None:
    records = [
        _episode(
            "object",
            0.04,
            0.02,
            graph_quality=index != 0,
        )
        for index in range(6)
    ]
    config = PrefixKinematicsDiagnosticConfig(
        minimum_common_episode_count=6,
        minimum_relative_improvement=0.05,
        minimum_win_fraction=0.60,
    )
    decision = build_source_decision(records, config=config)
    assert decision["policy_summaries"]["graph_harmonic_contact_v1"][
        "quality_valid_episode_count"
    ] == 5
    assert decision["passed"] is False


def test_result_validation_enforces_policy_and_target_boundaries() -> None:
    record = _episode("object", 0.04, 0.03)
    record["episode_id"] = "object/episode_0000"
    record["information_boundary"] = {
        "source_episode_only": True,
        "target_prefix_read": False,
        "target_future_read": False,
    }
    payload = {
        "schema_version": PREFIX_KINEMATICS_DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_kind": PREFIX_KINEMATICS_DIAGNOSTIC_KIND,
        "episode_records": [record],
        "information_boundary": {
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
            "registered_replication_result_changed": False,
        },
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    validate_source_prefix_kinematics_diagnostic(payload)

    payload["information_boundary"]["target_prefix_read"] = True
    payload["result_sha256"] = _canonical_sha256(payload)
    with pytest.raises(ValueError, match="boundary"):
        validate_source_prefix_kinematics_diagnostic(payload)


def test_policy_order_is_fixed() -> None:
    assert PREFIX_KINEMATICS_POLICIES == (
        "zero_v1",
        "global_contact_translation_v1",
        "graph_harmonic_contact_v1",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"baseline_chamfer_tolerance_m": -1.0}, "baseline_chamfer"),
        ({"minimum_relative_improvement": 1.0}, "below one"),
        ({"minimum_win_fraction": 1.1}, "at most one"),
        ({"minimum_common_episode_count": True}, "nonnegative integer"),
    ],
)
def test_diagnostic_config_rejects_invalid_controls(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PrefixKinematicsDiagnosticConfig(**kwargs)
