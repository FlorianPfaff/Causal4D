from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.causal4d_provider_v2 import (
    InitialReplayRequestV1,
    ReplayTrajectoryV1,
    RestartReplayRequestV1,
)
from causal4d.provider_contract import PhysicalBeliefProviderManifest
from causal4d.replay_provider_contract import (
    BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_CAPABILITIES,
    load_bayesian_phystwin_replay_provider_manifest,
    require_bayesian_phystwin_replay_provider,
    stable_replay_identifier,
    validate_bayesian_phystwin_replay_provider,
    validate_replay_trajectory,
)


def _manifest(**overrides) -> PhysicalBeliefProviderManifest:
    values = {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "a" * 40,
        "schema_version": 2,
        "capabilities": BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_CAPABILITIES,
        "artifact_schema_versions": {
            "GraphBelief": 1,
            "TwinBelief": 1,
            "ReplayRequest": 1,
            "ReplayTrajectory": 1,
        },
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_provider_v2",
            "provider_api_version": 2,
            "legacy_provider_api": "bayesian_phystwin.causal4d_provider_v1",
        },
    }
    values.update(overrides)
    return PhysicalBeliefProviderManifest(**values)


def test_installed_replay_provider_v2_is_compatible() -> None:
    manifest = load_bayesian_phystwin_replay_provider_manifest(
        provider_revision="replay-contract-test"
    )
    result = validate_bayesian_phystwin_replay_provider(manifest)

    assert result.compatible, result.as_dict()
    assert manifest.schema_version == 2
    assert (
        require_bayesian_phystwin_replay_provider(
            provider_revision="replay-contract-test"
        ).manifest_id
        == manifest.manifest_id
    )


def test_replay_provider_fails_closed_on_boundary_drift() -> None:
    missing = _manifest(
        capabilities=tuple(
            value
            for value in BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_CAPABILITIES
            if value != "stateless_replay_requests"
        )
    )
    assert not validate_bayesian_phystwin_replay_provider(missing).compatible

    with pytest.raises(ValueError, match="causal4d_provider_v2"):
        validate_bayesian_phystwin_replay_provider(
            _manifest(metadata={"provider_api": "wrong", "provider_api_version": 2})
        )


def test_replay_trajectory_is_bound_to_request_and_frame_provenance() -> None:
    request = InitialReplayRequestV1(
        request_id="request-v1",
        simulator_configuration_id="configuration-v1",
        initial_state_id="released-state-v1",
        group_log_scales=np.zeros(2),
        controller_points_m=np.zeros((4, 1, 3)),
        frame_count=3,
    )
    replay = ReplayTrajectoryV1(
        positions_m=np.zeros((3, 2, 3)),
        velocities_mps=np.ones((3, 2, 3)),
        frame_ids=np.arange(3),
        dt_s=0.03,
        request_id=request.request_id,
        simulator_configuration_id=request.simulator_configuration_id,
        initial_state_id=request.initial_state_id,
    )

    validate_replay_trajectory(request, replay, expected_dt_s=0.03)

    with pytest.raises(ValueError, match="request_id"):
        validate_replay_trajectory(
            request,
            replace(replay, request_id="other-request"),
            expected_dt_s=0.03,
        )
    with pytest.raises(ValueError, match="frame provenance"):
        validate_replay_trajectory(
            request,
            replace(replay, frame_ids=np.asarray([0, 2, 3])),
            expected_dt_s=0.03,
        )


def test_replay_identifier_is_stable_and_tamper_sensitive() -> None:
    first = stable_replay_identifier("request", {"frame": 3, "mode": "restart"})
    same = stable_replay_identifier("request", {"mode": "restart", "frame": 3})
    changed = stable_replay_identifier("request", {"frame": 4, "mode": "restart"})

    assert first == same
    assert first != changed
    assert first.startswith("request:")


def test_restart_replay_trajectory_uses_absolute_frame_provenance() -> None:
    request = RestartReplayRequestV1(
        request_id="restart-v1",
        simulator_configuration_id="configuration-v1",
        initial_state_id="endpoint-v1",
        group_log_scales=np.zeros(2),
        controller_points_m=np.zeros((8, 1, 3)),
        position_m=np.zeros((2, 3)),
        velocity_mps=np.ones((2, 3)),
        start_frame=5,
        stop_frame=8,
    )
    replay = ReplayTrajectoryV1(
        positions_m=np.zeros((3, 2, 3)),
        velocities_mps=np.ones((3, 2, 3)),
        frame_ids=np.asarray([5, 6, 7]),
        dt_s=0.03,
        request_id=request.request_id,
        simulator_configuration_id=request.simulator_configuration_id,
        initial_state_id=request.initial_state_id,
    )

    validate_replay_trajectory(request, replay, expected_dt_s=0.03)

    with pytest.raises(ValueError, match="timestep"):
        validate_replay_trajectory(request, replay, expected_dt_s=0.04)


def test_replay_identifier_rejects_invalid_identity_inputs() -> None:
    with pytest.raises(TypeError, match="namespace"):
        stable_replay_identifier(None, {"frame": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite JSON"):
        stable_replay_identifier("request", {"value": np.nan})
