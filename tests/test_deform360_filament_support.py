from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d_public import deform360_filament_support as support
from causal4d_public.deform360_filament_support import (
    FILAMENT_SUPPORT_CONFIG_KIND,
    FILAMENT_SUPPORT_KIND,
    FILAMENT_SUPPORT_POLICIES,
    FILAMENT_SUPPORT_SCHEMA_VERSION,
    FilamentSupportConfig,
    build_filament_support_decision,
    filament_support_config_sha256,
    filament_support_result_sha256,
    load_filament_support_lock,
    validate_source_filament_support_diagnostic,
)
from causal4d_public.deform360_replication_graph import (
    build_filament_sparse_graph,
    build_filament_sparse_graph_component_bridge,
)
from causal4d_public.deform360_rope_graph import (
    RopeCenterlineConfig,
    extract_rope_centerline,
    extract_rope_centerline_component_bridge,
)


def _connected_filament() -> np.ndarray:
    parameter = np.linspace(-1.0, 1.0, 120)
    curve = np.column_stack(
        (
            0.22 * parameter,
            0.06 * parameter**2,
            0.025 * np.sin(np.pi * parameter),
        )
    )
    rng = np.random.default_rng(35)
    repeated = np.repeat(curve, 10, axis=0)
    return repeated + rng.normal(scale=0.0015, size=repeated.shape)


def _disconnected_filament() -> np.ndarray:
    first_x = np.linspace(-0.30, -0.06, 90)
    second_x = np.linspace(0.06, 0.30, 90)
    first = np.column_stack(
        (first_x, 0.008 * np.sin(20.0 * first_x), np.zeros_like(first_x))
    )
    second = np.column_stack(
        (second_x, 0.008 * np.sin(20.0 * second_x), np.zeros_like(second_x))
    )
    return np.concatenate((first, second), axis=0)


def _metrics(
    content_id: str,
    *,
    p95: float,
    length: float,
    bridge: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "graph_content_sha256": content_id,
        "node_count": 21,
        "spring_count": 39,
        "centerline_length_m": length,
        "edge_length_coefficient_of_variation": 0.01,
        "point_to_centerline_node_distance_m": {
            "median": 0.005,
            "p95": p95,
            "maximum": 0.02,
        },
        "symmetric_chamfer_m": 0.01,
        "connectivity_policy": (
            "component-bridge-v1" if bridge is not None else "registered_v1"
        ),
        "component_bridge": bridge,
    }


def _boundary() -> dict[str, bool]:
    return {
        "source_episode_only": True,
        "reset_selection_uses_availability_only": True,
        "current_reset_hull_only_used_for_graph_construction": True,
        "future_geometry_used_for_graph_construction": False,
        "mechanics_scores_read": False,
        "calibration_outcomes_read": False,
        "target_prefix_read": False,
        "target_future_read": False,
    }


def _common_reset(
    episode_id: str,
    ordinal: int,
    *,
    content_id: str,
) -> dict[str, object]:
    metrics = _metrics(content_id, p95=0.01, length=0.4, bridge=None)
    return {
        "object_id": "rope",
        "episode_id": episode_id,
        "source_grid_path": f"source/{episode_id}.json",
        "source_grid_result_sha256": "a" * 64,
        "reset_ordinal": ordinal,
        "reset_hull_position": ordinal,
        "reset_raw_frame": 10 + ordinal,
        "available_hull_count": 20,
        "registered": {"status": "completed", "metrics": metrics},
        "primary": {"status": "completed", "metrics": dict(metrics)},
        "common_case_exact_graph_parity": True,
        "information_boundary": _boundary(),
    }


def _repaired_reset() -> dict[str, object]:
    message = "rope point cloud remains disconnected at the maximum neighbor count"
    bridge = {
        "component_count_before_bridge": 2,
        "maximum_neighbor_count_used": 24,
        "bridge_count": 1,
        "bridge_edges": [[10, 11]],
        "bridge_lengths_m": [0.04],
        "maximum_bridge_length_m": 0.04,
        "total_bridge_length_m": 0.04,
        "local_neighbor_edge_scale_m": 0.01,
        "maximum_bridge_to_local_scale_ratio": 4.0,
        "maximum_bridge_fraction_of_centerline_length": 0.1,
        "total_bridge_fraction_of_centerline_length": 0.1,
    }
    return {
        "object_id": "rope",
        "episode_id": "rope/episode_0002",
        "source_grid_path": "source/rope/episode_0002.json",
        "source_grid_result_sha256": "b" * 64,
        "reset_ordinal": 2,
        "reset_hull_position": 4,
        "reset_raw_frame": 12,
        "available_hull_count": 20,
        "registered": {
            "status": "technical_failure",
            "technical_failure": {
                "stage": "build_graph",
                "exception_type": "ValueError",
                "message": message,
            },
        },
        "primary": {
            "status": "completed",
            "metrics": _metrics("c" * 64, p95=0.012, length=0.42, bridge=bridge),
        },
        "common_case_exact_graph_parity": None,
        "information_boundary": _boundary(),
    }


def _small_config() -> FilamentSupportConfig:
    return FilamentSupportConfig(
        expected_total_reset_count=3,
        expected_registered_success_reset_count=2,
        expected_registered_failure_episode_count=1,
        expected_registered_failure_reset_count=1,
    )


def _prior_summary() -> dict[str, object]:
    return {
        "classification": "insufficient_common_episode_support",
        "technical_failure_episode_count": 1,
        "technical_failure_reset_count": 1,
        "technical_failures": [
            {
                "episode_id": "rope/episode_0002",
                "reset_ordinal": 2,
                "reset_raw_frame": 12,
                "message": (
                    "rope point cloud remains disconnected at the maximum "
                    "neighbor count"
                ),
            }
        ],
    }


def _records() -> list[dict[str, object]]:
    return [
        _common_reset("rope/episode_0000", 0, content_id="d" * 64),
        _common_reset("rope/episode_0001", 1, content_id="e" * 64),
        _repaired_reset(),
    ]


def _validation_payload() -> dict[str, object]:
    config = _small_config()
    prior = _prior_summary()
    records = _records()
    payload: dict[str, object] = {
        "schema_version": FILAMENT_SUPPORT_SCHEMA_VERSION,
        "artifact_kind": FILAMENT_SUPPORT_KIND,
        "config": config.as_dict(),
        "prior_reset_mechanics": prior,
        "reset_records": records,
        "decision": build_filament_support_decision(
            records,
            prior,
            config=config,
        ),
        "information_boundary": {
            "source_candidate_outcomes_read_for_identity_only": True,
            "source_reset_geometry_read_for_structure_scoring": True,
            "source_future_mechanics_outcomes_read": False,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
            "registered_reset_result_changed": False,
            "registered_replication_result_changed": False,
            "registered_36_execution_method_changed": False,
        },
    }
    payload["result_sha256"] = filament_support_result_sha256(payload)
    return payload


def test_component_bridge_preserves_connected_centerline_exactly() -> None:
    points = _connected_filament()
    config = RopeCenterlineConfig(node_count=21)

    registered, registered_diagnostics = extract_rope_centerline(
        points,
        config=config,
    )
    candidate, candidate_diagnostics = extract_rope_centerline_component_bridge(
        points,
        config=config,
    )

    np.testing.assert_array_equal(candidate, registered)
    assert candidate_diagnostics["connectivity_policy"] == "registered-parity"
    assert candidate_diagnostics["component_bridge"]["bridge_count"] == 0
    assert (
        candidate_diagnostics["centerline_length_m"]
        == registered_diagnostics["centerline_length_m"]
    )


def test_component_bridge_repairs_only_the_registered_disconnection() -> None:
    points = _disconnected_filament()
    with pytest.raises(ValueError, match="maximum neighbor count"):
        extract_rope_centerline(points)

    centerline, diagnostics = extract_rope_centerline_component_bridge(points)

    assert centerline.shape == (21, 3)
    assert diagnostics["connectivity_policy"] == "component-bridge-v1"
    bridge = diagnostics["component_bridge"]
    assert bridge["component_count_before_bridge"] == 2
    assert bridge["bridge_count"] == 1
    assert bridge["maximum_bridge_to_local_scale_ratio"] > 1.0
    assert bridge["total_bridge_fraction_of_centerline_length"] > 0.0


def test_component_bridge_does_not_hide_other_input_errors() -> None:
    with pytest.raises(ValueError, match="shape"):
        extract_rope_centerline_component_bridge(np.zeros((21, 2)))


def test_component_bridge_graph_is_exact_on_registered_support() -> None:
    points = _connected_filament()
    registered = build_filament_sparse_graph(points)
    candidate = build_filament_sparse_graph_component_bridge(points)

    np.testing.assert_array_equal(candidate.positions_m, registered.positions_m)
    np.testing.assert_array_equal(candidate.spring_edges, registered.spring_edges)
    np.testing.assert_array_equal(
        candidate.spring_families,
        registered.spring_families,
    )
    np.testing.assert_array_equal(candidate.masses, registered.masses)


def test_component_bridge_graph_completes_disconnected_support() -> None:
    graph = build_filament_sparse_graph_component_bridge(_disconnected_filament())

    assert graph.positions_m.shape == (21, 3)
    assert graph.diagnostics["centerline"]["component_bridge"]["bridge_count"] == 1


def test_locked_config_round_trip_and_target_boundary(tmp_path: Path) -> None:
    controls = {
        **FilamentSupportConfig().as_dict(),
        "selected_object_ids": ["002-rope-silk", "081-stripe-rope"],
        "policy_order": list(FILAMENT_SUPPORT_POLICIES),
        "registered_policy": "registered_v1",
        "primary_policy": "component_bridge_v1",
        "reset_selection": (
            "availability_only_evenly_spaced_including_prefix_and_latest_eligible"
        ),
        "primary_construction": (
            "maximum_knn_component_mst_bridge_then_registered_refinement_v1"
        ),
    }
    payload: dict[str, object] = {
        "artifact_kind": FILAMENT_SUPPORT_CONFIG_KIND,
        "config": controls,
        "information_boundary": {
            "source_only": True,
            "source_future_geometry_allowed_for_structure_scoring": True,
            "source_candidate_outcomes_allowed_for_identity_only": True,
            "calibration_outcomes_allowed": False,
            "target_prefix_allowed": False,
            "target_future_allowed": False,
            "registered_reset_result_mutable": False,
            "registered_method_mutable": False,
        },
        "prior_reset_mechanics": {},
        "protocol_config_sha256": "a" * 64,
        "required_parent_commit": "b" * 40,
        "schema_version": FILAMENT_SUPPORT_SCHEMA_VERSION,
        "source_milestone_manifest_sha256": "c" * 64,
    }
    payload["config_sha256"] = filament_support_config_sha256(payload)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    lock = load_filament_support_lock(path)
    assert lock["selected_object_ids"] == (
        "002-rope-silk",
        "081-stripe-rope",
    )
    assert lock["required_parent_commit"] == "b" * 40

    payload["information_boundary"]["target_prefix_allowed"] = True
    payload["config_sha256"] = filament_support_config_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden information"):
        load_filament_support_lock(path)


def test_decision_admits_exact_parity_and_bounded_repairs() -> None:
    decision = build_filament_support_decision(
        _records(),
        _prior_summary(),
        config=_small_config(),
    )

    assert decision["classification"] == "component_bridge_filament_support_admitted"
    assert decision["passed"] is True
    assert decision["exact_common_case_parity_count"] == 2
    assert decision["repaired_reset_count"] == 1
    assert decision["mechanics_rescoring_permitted"] is False


def test_decision_rejects_nonlocal_bridge_without_retuning() -> None:
    records = _records()
    records[2]["primary"]["metrics"]["component_bridge"][
        "maximum_bridge_to_local_scale_ratio"
    ] = 13.0

    decision = build_filament_support_decision(
        records,
        _prior_summary(),
        config=_small_config(),
    )

    assert decision["classification"] == "component_bridge_nonlocal_structure_failure"
    assert decision["passed"] is False


def test_result_validation_detects_parity_and_target_tampering() -> None:
    payload = _validation_payload()
    validate_source_filament_support_diagnostic(payload)

    payload["reset_records"][0]["common_case_exact_graph_parity"] = False
    payload["result_sha256"] = filament_support_result_sha256(payload)
    with pytest.raises(ValueError, match="parity accounting"):
        validate_source_filament_support_diagnostic(payload)

    payload = _validation_payload()
    payload["information_boundary"]["target_prefix_read"] = True
    payload["result_sha256"] = filament_support_result_sha256(payload)
    with pytest.raises(ValueError, match="information or method boundary"):
        validate_source_filament_support_diagnostic(payload)


def test_checked_in_lock_binds_the_frozen_reset_summary() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = load_filament_support_lock(
        root / "configs/causal4d_public/deform360_filament_support_v1.json"
    )
    summary = support._load_reset_summary(root, lock["payload"])

    assert summary["classification"] == "insufficient_common_episode_support"
    assert summary["technical_failure_episode_count"] == 7
    assert summary["technical_failure_reset_count"] == 8
