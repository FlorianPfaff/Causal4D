from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import causal4d.bpt_belief as belief_module
from bayesian_phystwin.causal4d_provider_v2 import (
    InitialReplayRequestV1,
    ReplayTrajectoryV1,
)

from causal4d.bpt_belief import (
    BPTBeliefExportConfig,
    build_twin_belief_from_replays,
    export_official_phystwin_twin_belief,
    lift_isotropic_discrepancy_variance,
)
from causal4d.contracts import build_causal_context


def _inputs():
    frame_count = 8
    train_end = 5
    observed = np.zeros((frame_count, 3, 3), dtype=float)
    observed[:, :, 0] = np.arange(frame_count)[:, None] * 0.01
    actions = np.zeros((frame_count, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="belief_test",
        case_id="synthetic",
        observations=observed,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=train_end,
    )
    replay = np.zeros((2, frame_count, 5, 3), dtype=float)
    replay[0, :, :3] = observed
    replay[1, :, :3] = observed
    replay[1, :, :, 0] -= np.arange(frame_count)[:, None] * 0.001
    replay[:, :, 3, 0] = replay[:, :, 0, 0]
    replay[:, :, 4, 0] = replay[:, :, 2, 0]
    velocity = np.zeros_like(replay)
    velocity[:, 1:] = np.diff(replay, axis=1) / 0.03
    valid = np.ones((frame_count, 3), dtype=bool)
    return context, replay, velocity, observed, valid


def _belief(context, replay, velocity, observed, valid):
    return build_twin_belief_from_replays(
        context=context,
        replay_positions_m=replay,
        replay_velocities_mps=velocity,
        observed_positions_m=observed,
        observed_valid=valid,
        theta=np.asarray([[0.0, 0.0], [0.2, -0.1]]),
        theta_names=("object", "controller"),
        weights=np.asarray([0.7, 0.3]),
        config=BPTBeliefExportConfig(interpolation_neighbors=2),
    )


def test_full_belief_uses_particle_specific_endpoint_state() -> None:
    context, replay, velocity, observed, valid = _inputs()
    belief = _belief(context, replay, velocity, observed, valid)
    assert np.array_equal(belief.endpoint_position_m, replay[:, 4])
    assert np.array_equal(belief.endpoint_velocity_mps, velocity[:, 4])
    assert not np.array_equal(
        belief.endpoint_position_m[0], belief.endpoint_position_m[1]
    )
    assert belief.metadata["future_frames_read_by_estimator"] == 0
    assert belief.metadata["maximum_pairwise_endpoint_rmse_m"] > 0.0


def test_belief_estimation_cannot_see_changed_future_frames() -> None:
    context, replay, velocity, observed, valid = _inputs()
    first = _belief(context, replay, velocity, observed, valid)
    changed_replay = replay.copy()
    changed_velocity = velocity.copy()
    changed_observed = observed.copy()
    changed_valid = valid.copy()
    changed_replay[:, 5:] += 1000.0
    changed_velocity[:, 5:] -= 1000.0
    changed_observed[5:] = -1000.0
    changed_valid[5:] = False
    second = _belief(
        context,
        changed_replay,
        changed_velocity,
        changed_observed,
        changed_valid,
    )
    assert first.artifact_id == second.artifact_id
    assert np.array_equal(first.endpoint_position_m, second.endpoint_position_m)
    assert np.array_equal(first.discrepancy_mean_m, second.discrepancy_mean_m)
    assert np.array_equal(
        first.discrepancy_variance_m2,
        second.discrepancy_variance_m2,
    )


def test_discrepancy_is_separate_from_the_replayed_state() -> None:
    context, replay, velocity, observed, valid = _inputs()
    belief = _belief(context, replay, velocity, observed, valid)
    assert np.array_equal(belief.endpoint_position_m, replay[:, 4])
    assert np.linalg.norm(belief.discrepancy_mean_m[1]) > 0.0
    assert "not injected" in belief.metadata["discrepancy_role"]


def test_variance_lift_uses_squared_interpolation_weights() -> None:
    tracked = np.asarray([1.0, 4.0])
    indices = np.asarray([[0, 1]])
    weights = np.asarray([[0.25, 0.75]])
    lifted = lift_isotropic_discrepancy_variance(
        tracked,
        state_count=3,
        indices=indices,
        weights=weights,
    )
    expected_extra = 0.25**2 * 1.0 + 0.75**2 * 4.0
    assert np.allclose(lifted[:2, 0], tracked)
    assert np.allclose(lifted[2], expected_extra)


def test_official_export_uses_explicit_initial_replay_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_count = 4
    train_end = 3
    controls = np.zeros((frame_count, 1, 3), dtype=np.float32)
    observed = np.zeros((frame_count, 1, 3), dtype=np.float32)
    context = build_causal_context(
        protocol_id="initial-replay-v2",
        case_id="unit",
        observations=observed,
        observed_actions=controls,
        counterfactual_actions=controls,
        intervention_frame=train_end,
    )
    particles = SimpleNamespace(
        log_scales=np.asarray([[0.0, 0.0], [0.2, -0.1]], dtype=float),
        grid_indices=np.asarray([[0, 0], [1, 0]], dtype=int),
        weights=np.asarray([0.6, 0.4], dtype=float),
        source_weight_key="posterior_weights",
        selection_method="top_mass",
        retained_probability_mass=1.0,
        represented_probability_mass=1.0,
    )
    backend = SimpleNamespace(
        case_name="unit",
        train_end_frame=train_end,
        config=SimpleNamespace(
            self_collision=False,
            dt=0.01,
            num_substeps=2,
            deterministic_spring_forces=True,
            device="cpu",
        ),
        official_repo=tmp_path / "official",
        data={},
        optimal={},
        checkpoint_path=tmp_path / "checkpoint.pt",
        graph=SimpleNamespace(),
        original_count=1,
        surface_points=np.zeros((0, 3), dtype=np.float32),
        particles=particles,
        controller_points=controls,
        visible=np.ones((frame_count, 1), dtype=bool),
        motion_valid=np.ones((frame_count, 1), dtype=bool),
        object_points=observed,
        profile_path=tmp_path / "profile.npz",
        frame_dt_s=0.02,
    )
    backend.replay_simulator_configuration_id = lambda graph: "configuration-v2"
    backend.replay_released_initial_state_id = lambda: "released-state-v1"
    backend.default_manifest = lambda: {
        "replay_provider": {"schema_version": 2},
        "replay_contract": {"frame_dt_s": 0.02},
    }
    requests: list[InitialReplayRequestV1] = []
    closed = False

    class FakeProvider:
        device = "cpu"
        frame_dt_s = 0.02
        simulator_configuration_id = "configuration-v2"
        released_initial_state_id = "released-state-v1"

        def replay(self, request: InitialReplayRequestV1) -> ReplayTrajectoryV1:
            requests.append(request)
            value = float(np.sum(request.group_log_scales))
            positions = np.full((request.frame_count, 1, 3), value, dtype=np.float32)
            velocities = np.full(
                (request.frame_count, 1, 3),
                value + 1.0,
                dtype=np.float32,
            )
            return ReplayTrajectoryV1(
                positions_m=positions,
                velocities_mps=velocities,
                frame_ids=np.arange(request.frame_count),
                dt_s=self.frame_dt_s,
                request_id=request.request_id,
                simulator_configuration_id=request.simulator_configuration_id,
                initial_state_id=request.initial_state_id,
            )

        def close(self) -> None:
            nonlocal closed
            closed = True

    def factory(*args: Any, **kwargs: Any) -> FakeProvider:
        del args, kwargs
        return FakeProvider()

    monkeypatch.setattr(belief_module, "create_official_replay_provider", factory)
    belief = export_official_phystwin_twin_belief(
        backend,
        context=context,
        config=BPTBeliefExportConfig(interpolation_neighbors=1),
    )

    assert closed
    assert len(requests) == 2
    for index, request in enumerate(requests):
        assert request.frame_count == train_end
        np.testing.assert_array_equal(request.controller_points_m, controls)
        np.testing.assert_allclose(
            request.group_log_scales,
            particles.log_scales[index],
        )
        assert request.simulator_configuration_id == "configuration-v2"
        assert request.initial_state_id == "released-state-v1"
    assert belief.metadata["replay_provider_api_version"] == 2
    assert len(belief.metadata["initial_replay_requests"]) == 2
    np.testing.assert_allclose(
        belief.endpoint_velocity_mps[:, 0, 0],
        [1.0, 1.1],
    )
