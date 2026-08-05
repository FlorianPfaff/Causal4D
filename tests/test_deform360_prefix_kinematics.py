from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from causal4d_public.deform360_prefix_kinematics import (
    PREFIX_KINEMATICS_POLICIES,
    PrefixKinematicsConfig,
    build_prefix_velocity_policies,
    clip_velocity_norms,
    controller_patch_velocities_from_prefix,
    global_contact_translation_velocity,
    graph_harmonic_contact_velocity,
    recent_contact_mask,
)
from causal4d_public.deform360_replication_graph import Deform360SparseGraph


def _chain_graph() -> Deform360SparseGraph:
    positions = np.column_stack((np.linspace(0.0, 0.4, 5), np.zeros(5), np.zeros(5)))
    return Deform360SparseGraph(
        positions_m=positions,
        spring_edges=np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]]),
        spring_families=np.zeros(4, dtype=np.int8),
        masses=np.ones(5),
        stratum="filament",
        diagnostics={},
    )


def _disconnected_graph() -> Deform360SparseGraph:
    positions = np.column_stack((np.linspace(0.0, 0.4, 5), np.zeros(5), np.zeros(5)))
    return Deform360SparseGraph(
        positions_m=positions,
        spring_edges=np.asarray([[0, 1], [1, 2], [0, 2], [3, 4]]),
        spring_families=np.zeros(4, dtype=np.int8),
        masses=np.ones(5),
        stratum="filament",
        diagnostics={},
    )


def _install_fake_deform360(
    monkeypatch: pytest.MonkeyPatch,
) -> list[float]:
    transforms = np.repeat(np.eye(4)[None, None], 8, axis=0)
    transforms[:, 0, 0, 3] = np.arange(8) * 0.03
    transforms[7] = np.nan
    state = SimpleNamespace(
        openings=np.zeros((8, 1)),
        T_worlds=transforms,
    )
    called_x: list[float] = []

    def load_robot_state(_path):
        return state

    def gripper_taxel_points(_opening: float, transform: np.ndarray) -> np.ndarray:
        translation = np.asarray(transform[:3, 3], dtype=float)
        called_x.append(float(translation[0]))
        return np.asarray(
            [
                translation,
                translation + np.asarray([0.01, 0.0, 0.0]),
                translation + np.asarray([0.02, 0.0, 0.0]),
            ]
        )

    deform360 = ModuleType("deform360")
    processing = ModuleType("deform360.processing")
    control = ModuleType("deform360.processing.control_points_stage")
    robot = ModuleType("deform360.robot")
    control.gripper_taxel_points = gripper_taxel_points  # type: ignore[attr-defined]
    robot.load_robot_state = load_robot_state  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deform360", deform360)
    monkeypatch.setitem(sys.modules, "deform360.processing", processing)
    monkeypatch.setitem(
        sys.modules,
        "deform360.processing.control_points_stage",
        control,
    )
    monkeypatch.setitem(sys.modules, "deform360.robot", robot)
    return called_x


def test_recent_contact_mask_uses_only_prefix_tail() -> None:
    active = np.zeros((9, 2), dtype=bool)
    active[3, 0] = True
    active[8, 1] = True
    mask = recent_contact_mask(
        active,
        prefix_endpoint_frame=5,
        memory_frames=3,
    )
    assert mask.tolist() == [True, False]
    assert not mask.flags.writeable


def test_controller_patch_velocity_is_causal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_x = _install_fake_deform360(monkeypatch)
    velocity = controller_patch_velocities_from_prefix(
        tmp_path,
        prefix_endpoint_frame=6,
        contact_associations=(
            {
                "robot_axis": 0,
                "selected_taxel_indices": [0, 1],
                "contact_offset_m": [0.0, 0.0, 0.0],
            },
        ),
        dt_seconds=0.1,
        lookback_frames=3,
        maximum_speed_m_s=2.0,
    )
    np.testing.assert_allclose(velocity, [[0.3, 0.0, 0.0]], atol=1e-12)
    assert called_x == pytest.approx([0.09, 0.18])
    assert all(np.isfinite(called_x))
    assert not velocity.flags.writeable


@pytest.mark.parametrize(
    ("association", "message"),
    [
        (
            {
                "robot_axis": 0,
                "selected_taxel_indices": None,
                "contact_offset_m": [0.0, 0.0, 0.0],
            },
            "taxel indices",
        ),
        (
            {
                "robot_axis": 0,
                "selected_taxel_indices": [0],
                "contact_offset_m": None,
            },
            "three-element sequence",
        ),
        (
            {
                "robot_axis": 0,
                "selected_taxel_indices": [0, 0],
                "contact_offset_m": [0.0, 0.0, 0.0],
            },
            "repeats a taxel index",
        ),
    ],
)
def test_controller_patch_velocity_rejects_malformed_contact_associations(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    association: dict[str, object],
    message: str,
) -> None:
    _install_fake_deform360(monkeypatch)
    with pytest.raises(ValueError, match=message):
        controller_patch_velocities_from_prefix(
            tmp_path,
            prefix_endpoint_frame=6,
            contact_associations=(association,),
            dt_seconds=0.1,
            lookback_frames=3,
            maximum_speed_m_s=2.0,
        )


def test_global_policy_applies_mean_active_controller_velocity() -> None:
    graph = _chain_graph()
    field = global_contact_translation_velocity(
        graph,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        np.asarray([True, True]),
        maximum_node_speed_m_s=3.0,
    )
    np.testing.assert_allclose(field, np.repeat([[0.5, 1.0, 0.0]], 5, axis=0))


def test_graph_policy_rejects_disconnected_stretch_graph() -> None:
    with pytest.raises(ValueError, match="disconnected"):
        graph_harmonic_contact_velocity(
            _disconnected_graph(),
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([True]),
            (0,),
            contact_weight=100.0,
            smoothness_weight=1.0,
            ridge_weight=0.1,
            maximum_node_speed_m_s=2.0,
        )


def test_graph_policy_diffuses_contact_velocity() -> None:
    graph = _chain_graph()
    field = graph_harmonic_contact_velocity(
        graph,
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([True]),
        (0,),
        contact_weight=100.0,
        smoothness_weight=1.0,
        ridge_weight=0.1,
        maximum_node_speed_m_s=2.0,
    )
    assert np.all(field[:, 0] > 0.0)
    assert np.all(np.diff(field[:, 0]) < 0.0)
    np.testing.assert_allclose(field[:, 1:], 0.0, atol=1e-12)
    assert field[0, 0] < 1.0
    assert not field.flags.writeable


def test_graph_policy_combines_duplicate_contact_nodes() -> None:
    graph = _chain_graph()
    forward = graph_harmonic_contact_velocity(
        graph,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([True, True]),
        (2, 2),
        contact_weight=100.0,
        smoothness_weight=1.0,
        ridge_weight=0.1,
        maximum_node_speed_m_s=2.0,
    )
    reversed_field = graph_harmonic_contact_velocity(
        graph,
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([True, True]),
        (2, 2),
        contact_weight=100.0,
        smoothness_weight=1.0,
        ridge_weight=0.1,
        maximum_node_speed_m_s=2.0,
    )
    np.testing.assert_allclose(forward, reversed_field)
    assert forward[2, 0] == pytest.approx(forward[2, 1])


def test_no_recent_contact_preserves_zero_initial_velocity() -> None:
    graph = _chain_graph()
    field = graph_harmonic_contact_velocity(
        graph,
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([False]),
        (0,),
        contact_weight=100.0,
        smoothness_weight=1.0,
        ridge_weight=0.1,
        maximum_node_speed_m_s=2.0,
    )
    np.testing.assert_array_equal(field, np.zeros((5, 3)))


def test_velocity_clipping_is_radial() -> None:
    clipped = clip_velocity_norms(np.asarray([[3.0, 4.0, 0.0]]), 2.0)
    np.testing.assert_allclose(clipped, [[1.2, 1.6, 0.0]])


def test_build_policies_records_causal_boundary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_deform360(monkeypatch)
    graph = _chain_graph()
    schedule = np.zeros((8, 1), dtype=bool)
    schedule[5, 0] = True
    policies, diagnostics = build_prefix_velocity_policies(
        graph,
        tmp_path,
        prefix_endpoint_frame=6,
        contact_associations=(
            {
                "robot_axis": 0,
                "selected_taxel_indices": [0, 1],
                "contact_offset_m": [0.0, 0.0, 0.0],
            },
        ),
        full_contact_active=schedule,
        contact_node_indices=(0,),
        dt_seconds=0.1,
        config=PrefixKinematicsConfig(),
    )
    assert tuple(policies) == PREFIX_KINEMATICS_POLICIES
    np.testing.assert_array_equal(policies["zero_v1"], np.zeros((5, 3)))
    assert diagnostics["causal_start_frame"] == 3
    assert diagnostics["active_controller_indices"] == [0]
    assert diagnostics["future_robot_state_read"] is False
    assert diagnostics["future_object_geometry_read"] is False
    assert diagnostics["future_tactile_read"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lookback_frames": True}, "lookback_frames"),
        ({"contact_memory_frames": 0}, "contact_memory_frames"),
        ({"contact_weight": np.nan}, "contact_weight"),
        ({"maximum_node_speed_m_s": -1.0}, "maximum_node_speed_m_s"),
    ],
)
def test_config_rejects_invalid_controls(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PrefixKinematicsConfig(**kwargs)
