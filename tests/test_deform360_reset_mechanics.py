from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from causal4d_public import deform360_reset_mechanics as mechanics
from causal4d_public.deform360_reset_mechanics import (
    RESET_MECHANICS_CONFIG_KIND,
    RESET_MECHANICS_KIND,
    RESET_MECHANICS_SCHEMA_VERSION,
    ResetMechanicsConfig,
    build_reset_mechanics_decision,
    load_reset_mechanics_lock,
    reset_mechanics_config_sha256,
    select_reset_positions,
    summarize_reset_horizon,
    validate_source_reset_mechanics_diagnostic,
)


def _episode(
    episode_id: str,
    *,
    candidate: tuple[float, float, float] = (0.8, 0.7, 0.6),
    persistence: tuple[float, float, float] = (1.0, 1.0, 1.0),
    quality: tuple[bool, bool, bool] = (True, True, True),
    reproduction: bool = True,
) -> dict[str, object]:
    horizons = (1, 3, 6)
    resets = []
    for reset_index in range(3):
        reset_horizons = {}
        for horizon, score, baseline, valid in zip(
            horizons,
            candidate,
            persistence,
            quality,
            strict=True,
        ):
            reset_horizons[f"next_{horizon}_observations"] = {
                "finite": True,
                "mean_chamfer_m": score + 0.01 * reset_index,
                "persistence_mean_chamfer_m": baseline,
                "quality_valid": valid,
            }
        resets.append(
            {
                "reset_ordinal": reset_index,
                "reset_hull_position": reset_index * 2,
                "status": "completed",
                "horizons": reset_horizons,
            }
        )
    return {
        "object_id": episode_id.split("/", maxsplit=1)[0],
        "episode_id": episode_id,
        "resets": resets,
        "prefix_baseline_reproduction": {"passed": reproduction},
    }


def _complete_validation_episode(episode_id: str) -> dict[str, object]:
    frames = list(range(11))
    positions = select_reset_positions(
        frames,
        reset_count=3,
        maximum_horizon_observation_count=6,
    )
    resets = []
    for ordinal, position in enumerate(positions):
        resets.append(
            {
                "reset_ordinal": ordinal,
                "reset_hull_position": position,
                "status": "completed",
                "horizons": {
                    "next_1_observations": {"finite": True},
                    "next_3_observations": {"finite": True},
                    "next_6_observations": {"finite": True},
                },
            }
        )
    return {
        "object_id": episode_id.split("/", maxsplit=1)[0],
        "episode_id": episode_id,
        "raw_hull_frame_indices": frames,
        "reset_positions": list(positions),
        "resets": resets,
        "completed_reset_count": 3,
        "technical_failure_reset_count": 0,
        "technically_complete": True,
        "information_boundary": {
            "source_episode_only": True,
            "reset_selection_uses_availability_only": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
        },
    }


def _validation_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": RESET_MECHANICS_SCHEMA_VERSION,
        "artifact_kind": RESET_MECHANICS_KIND,
        "config": ResetMechanicsConfig().as_dict(),
        "episode_records": [_complete_validation_episode("rope/episode_0001")],
        "information_boundary": {
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
            "registered_replication_result_changed": False,
            "registered_36_execution_method_changed": False,
        },
    }
    payload["result_sha256"] = mechanics._artifact_sha256(payload)
    return payload


def test_reset_scoring_uses_raw_frame_gaps_and_registered_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ModuleType("causal4d_public.deform360_replication_warp")

    def score(reference: object, prediction: np.ndarray) -> dict[str, object]:
        values = np.asarray(prediction, dtype=np.float64)
        return {
            "mean_m": float(np.mean(np.abs(values[..., 0]))),
            "late_mean_m": float(np.mean(np.abs(values[..., 0]))),
            "per_frame_m": [0.0] * len(values),
        }

    def strain(graph: object, prediction: np.ndarray) -> dict[str, float]:
        del graph, prediction
        return {"p95": 0.1, "p99": 0.2, "maximum": 0.3}

    fake.sparse_trajectory_chamfer_m = score
    fake.sparse_graph_strain_summary = strain
    monkeypatch.setitem(
        sys.modules,
        "causal4d_public.deform360_replication_warp",
        fake,
    )
    graph = SimpleNamespace(positions_m=np.ones((2, 3), dtype=np.float64))
    observation = SimpleNamespace(
        raw_hull_frame_indices=np.asarray([10, 11, 13, 16, 18, 20, 22]),
        prefix_endpoint_frame=10,
        reference_hulls_m=tuple(np.zeros((2, 3)) for _ in range(7)),
        case=SimpleNamespace(graph=graph, dt_seconds=0.1),
    )
    prediction = np.zeros((13, 2, 3), dtype=np.float64)
    result = mechanics._score_reset_prediction(
        observation,
        prediction,
        maximum_p99_relative_edge_strain=0.25,
        horizons=(1, 3, 6),
    )
    assert tuple(result) == (
        "next_1_observations",
        "next_3_observations",
        "next_6_observations",
    )
    assert result["next_1_observations"]["horizon_raw_frame_gap"] == 1
    assert result["next_3_observations"]["horizon_raw_frame_gap"] == 6
    assert result["next_6_observations"]["horizon_raw_frame_gap"] == 12
    assert result["next_6_observations"]["horizon_seconds"] == pytest.approx(1.2)
    assert result["next_6_observations"]["quality_valid"] is True


def test_registered_reset_retains_graph_construction_failure() -> None:
    def fail_to_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("rope point cloud remains disconnected")

    frames = np.arange(7, dtype=np.int64)
    hulls = tuple(np.zeros((2, 3), dtype=np.float64) for _ in range(7))
    result = mechanics._evaluate_registered_reset(
        build_observation=fail_to_build,
        episode_dir=Path("/tmp/episode"),
        episode_id="rope/episode_0001",
        stratum="filament",
        frames=frames,
        hulls=hulls,
        schedule={},
        reset_ordinal=0,
        reset_position=0,
        official_phystwin_repo=Path("/tmp/phystwin"),
        simulation_config=object(),
        candidate=object(),
        device="cpu",
        horizons=(1, 3, 6),
    )
    assert result["status"] == "technical_failure"
    assert result["technical_failure"] == {
        "stage": "build_observation",
        "exception_type": "ValueError",
        "message": "rope point cloud remains disconnected",
    }
    assert "horizons" not in result


def test_reset_positions_depend_only_on_frame_availability() -> None:
    frames = [0, 2, 3, 7, 8, 9, 11, 15, 18, 20, 21]
    assert select_reset_positions(
        frames,
        reset_count=3,
        maximum_horizon_observation_count=6,
    ) == (0, 2, 4)

    translated = [frame + 100 for frame in frames]
    assert select_reset_positions(
        translated,
        reset_count=3,
        maximum_horizon_observation_count=6,
    ) == (0, 2, 4)


@pytest.mark.parametrize(
    "frames, reset_count, horizon",
    [
        ([0, 1, 2], 3, 1),
        ([0, 1, 1, 2, 3, 4, 5, 6], 2, 3),
        ([0.0, 1.0, 2.0, 3.0], 1, 1),
    ],
)
def test_reset_positions_reject_invalid_availability(
    frames: list[float | int],
    reset_count: int,
    horizon: int,
) -> None:
    with pytest.raises(ValueError):
        select_reset_positions(
            frames,
            reset_count=reset_count,
            maximum_horizon_observation_count=horizon,
        )


def test_horizon_summary_uses_episode_means_not_reset_pseudoreplication() -> None:
    config = ResetMechanicsConfig(
        minimum_common_episode_count=2,
        minimum_relative_improvement=0.05,
        minimum_win_fraction=0.5,
        minimum_quality_valid_fraction=0.8,
    )
    episodes = [
        _episode("rope/episode_0001"),
        _episode(
            "cloth/episode_0002",
            candidate=(0.9, 0.8, 0.7),
            quality=(True, True, False),
        ),
    ]
    summary = summarize_reset_horizon(episodes, 3, config=config)
    assert summary["common_episode_count"] == 2
    assert len(summary["episode_records"]) == 2
    assert summary["episode_win_fraction"] == 1.0
    assert summary["mean_quality_valid_fraction"] == 1.0
    assert summary["passed"] is True

    horizon_six = summarize_reset_horizon(episodes, 6, config=config)
    assert horizon_six["mean_quality_valid_fraction"] == 0.5
    assert horizon_six["passed"] is False


def test_incomplete_reset_excludes_the_complete_episode_unit() -> None:
    config = ResetMechanicsConfig(minimum_common_episode_count=1)
    episode = _episode("rope/episode_0001")
    episode["resets"][1]["horizons"]["next_3_observations"]["finite"] = False
    summary = summarize_reset_horizon([episode], 3, config=config)
    assert summary["common_episode_count"] == 0
    assert summary["episode_records"] == []
    assert summary["passed"] is False


def test_technical_failure_excludes_the_complete_episode_unit() -> None:
    config = ResetMechanicsConfig(minimum_common_episode_count=1)
    episode = _episode("rope/episode_0001")
    episode["resets"][1] = {
        "reset_ordinal": 1,
        "reset_hull_position": 2,
        "status": "technical_failure",
        "technical_failure": {
            "stage": "build_observation",
            "exception_type": "ValueError",
            "message": "disconnected graph",
        },
    }
    summary = summarize_reset_horizon([episode], 3, config=config)
    assert summary["common_episode_count"] == 0
    assert summary["excluded_episode_count"] == 1
    assert summary["technical_failure_episode_count"] == 1
    assert summary["passed"] is False


def test_decision_does_not_relabel_insufficient_support_as_mechanics_failure() -> None:
    config = ResetMechanicsConfig(
        minimum_common_episode_count=2,
        minimum_relative_improvement=0.0,
        minimum_win_fraction=0.0,
        minimum_quality_valid_fraction=0.0,
    )
    failed = _episode("cloth/episode_0002")
    failed["resets"][1] = {
        "reset_ordinal": 1,
        "reset_hull_position": 2,
        "status": "technical_failure",
        "technical_failure": {
            "stage": "build_observation",
            "exception_type": "ValueError",
            "message": "disconnected graph",
        },
    }
    decision = build_reset_mechanics_decision(
        [_episode("rope/episode_0001"), failed],
        config=config,
    )
    assert decision["baseline_reproduction_passed"] is True
    assert decision["technical_failure_episode_count"] == 1
    assert decision["technical_failure_reset_count"] == 1
    assert decision["classification"] == "insufficient_common_episode_support"
    assert decision["passed"] is False


def test_decision_identifies_the_first_failed_horizon() -> None:
    config = ResetMechanicsConfig(
        minimum_common_episode_count=2,
        minimum_relative_improvement=0.05,
        minimum_win_fraction=0.5,
        minimum_quality_valid_fraction=0.8,
    )
    episodes = [
        _episode("rope/episode_0001", candidate=(0.8, 0.8, 1.1)),
        _episode("cloth/episode_0002", candidate=(0.8, 0.8, 1.1)),
    ]
    decision = build_reset_mechanics_decision(episodes, config=config)
    assert decision["baseline_reproduction_passed"] is True
    assert decision["first_failed_horizon_observations"] == 6
    assert decision["classification"] == "multi_step_dynamics_accumulation_failure"
    assert decision["passed"] is False
    assert decision["registered_method_changed"] is False


def test_decision_fails_closed_when_the_prefix_baseline_does_not_reproduce() -> None:
    config = ResetMechanicsConfig(
        minimum_common_episode_count=1,
        minimum_relative_improvement=0.0,
        minimum_win_fraction=0.0,
        minimum_quality_valid_fraction=0.0,
    )
    decision = build_reset_mechanics_decision(
        [_episode("rope/episode_0001", reproduction=False)],
        config=config,
    )
    assert decision["classification"] == "baseline_reproduction_failure"
    assert decision["passed"] is False


def test_lock_round_trip_and_target_boundary(tmp_path: Path) -> None:
    payload: dict[str, object] = {
        "artifact_kind": RESET_MECHANICS_CONFIG_KIND,
        "config": {
            **ResetMechanicsConfig().as_dict(),
            "candidate_selection": (
                "quality_constrained_source_oracle_else_finite_oracle"
            ),
            "reset_selection": (
                "availability_only_evenly_spaced_including_prefix_and_latest_eligible"
            ),
            "initial_velocity_policy": "zero_v1",
            "selected_object_ids": ["rope"],
        },
        "information_boundary": {
            "source_only": True,
            "source_future_geometry_allowed_for_scoring": True,
            "source_tactile_allowed": True,
            "calibration_outcomes_allowed": False,
            "target_prefix_allowed": False,
            "target_future_allowed": False,
            "registered_replication_result_mutable": False,
        },
        "protocol_config_sha256": "a" * 64,
        "schema_version": 1,
        "source_backend_decision_result_sha256": "b" * 64,
        "source_milestone_manifest_sha256": "c" * 64,
    }
    payload["config_sha256"] = reset_mechanics_config_sha256(payload)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    lock = load_reset_mechanics_lock(path)
    assert lock["selected_object_ids"] == ("rope",)
    assert lock["config"] == ResetMechanicsConfig()

    payload["information_boundary"]["target_prefix_allowed"] = True
    payload["config_sha256"] = reset_mechanics_config_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden information"):
        load_reset_mechanics_lock(path)


def test_result_validation_retains_well_formed_technical_failure() -> None:
    payload = _validation_payload()
    episode = payload["episode_records"][0]
    episode["resets"][1] = {
        "reset_ordinal": 1,
        "reset_hull_position": 2,
        "status": "technical_failure",
        "technical_failure": {
            "stage": "build_observation",
            "exception_type": "ValueError",
            "message": "rope point cloud remains disconnected",
        },
    }
    episode["completed_reset_count"] = 2
    episode["technical_failure_reset_count"] = 1
    episode["technically_complete"] = False
    payload["result_sha256"] = mechanics._artifact_sha256(payload)
    validate_source_reset_mechanics_diagnostic(payload)

    episode["resets"][1]["horizons"] = {}
    payload["result_sha256"] = mechanics._artifact_sha256(payload)
    with pytest.raises(ValueError, match="contains scientific scores"):
        validate_source_reset_mechanics_diagnostic(payload)


def test_result_validation_detects_tampering_and_target_access() -> None:
    payload = _validation_payload()
    validate_source_reset_mechanics_diagnostic(payload)

    payload["episode_records"][0]["reset_positions"] = [0, 1, 4]
    payload["result_sha256"] = mechanics._artifact_sha256(payload)
    with pytest.raises(ValueError, match="availability-only reset ladder"):
        validate_source_reset_mechanics_diagnostic(payload)

    payload = _validation_payload()
    payload["information_boundary"]["target_prefix_read"] = True
    payload["result_sha256"] = mechanics._artifact_sha256(payload)
    with pytest.raises(ValueError, match="information or claim boundary"):
        validate_source_reset_mechanics_diagnostic(payload)
