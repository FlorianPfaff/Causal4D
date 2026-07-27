import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bayesian_phystwin.causal4d_provider_v2 import (
    PhysTwinReplayProvider,
    ReplayRequestV1,
    ReplayTrajectoryV1,
    RestartReplayRequestV1,
)
from causal4d.phystwin_resumable import _RolloutCacheSession
from causal4d.rollout_cache import ContentAddressedReplayCache


class FakeProvider:
    def __init__(self, counter: dict[str, int], **kwargs: Any) -> None:
        counter["providers"] += 1
        self.counter = counter
        self.device = str(kwargs["device"])
        self.frame_dt_s = float(kwargs["dt"]) * int(kwargs["num_substeps"])
        self.simulator_configuration_id = str(kwargs["simulator_configuration_id"])
        self.released_initial_state_id = str(kwargs["released_initial_state_id"])

    def replay(self, request: ReplayRequestV1) -> ReplayTrajectoryV1:
        if not isinstance(request, RestartReplayRequestV1):
            raise TypeError("unit provider expects a restart request")
        self.counter["replays"] += 1
        frames = request.stop_frame - request.start_frame
        positions = np.repeat(request.position_m[None], frames, axis=0)
        positions += float(np.sum(request.group_log_scales))
        velocities = np.repeat(request.velocity_mps[None], frames, axis=0)
        velocities += 0.25
        return ReplayTrajectoryV1(
            positions_m=positions,
            velocities_mps=velocities,
            frame_ids=np.arange(request.start_frame, request.stop_frame),
            dt_s=self.frame_dt_s,
            request_id=request.request_id,
            simulator_configuration_id=request.simulator_configuration_id,
            initial_state_id=request.initial_state_id,
        )

    def close(self) -> None:
        self.counter["closes"] += 1


def _graph() -> SimpleNamespace:
    return SimpleNamespace(
        vertices=np.zeros((3, 3), dtype=np.float32),
        springs=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        rest_lengths=np.ones(2, dtype=np.float32),
        masses=np.ones(3, dtype=np.float32),
        num_object_springs=2,
        num_object_points=3,
    )


def _session(tmp_path: Path) -> _RolloutCacheSession:
    return _RolloutCacheSession(
        cache=ContentAddressedReplayCache(tmp_path / "cache"),
        provider_manifest={
            "manifest_id": "provider-v2",
            "provider_version": "0.4.1",
            "provider_revision": "revision",
            "schema_version": 2,
        },
        provider_source={"fingerprint": "provider-source"},
        official_source={"fingerprint": "source", "revision": "source-rev"},
        numerical_runtime={"python_version": "unit"},
        source_artifacts_sha256={"checkpoint": "c" * 64},
        case_name="unit",
        twin_belief_id="belief",
    )


def _factory_arguments() -> tuple[tuple[Any, ...], dict[str, Any]]:
    return (
        ("official", {"data": True}, {"params": True}, "checkpoint", _graph()),
        {
            "num_surface_points": 3,
            "original_count": 3,
            "dt": 1e-4,
            "num_substeps": 4,
            "self_collision": False,
            "simulator_configuration_id": "configuration-v2",
            "released_initial_state_id": "released-state-v1",
            "deterministic_spring_forces": True,
            "spring_parameterization": "grouped",
            "device": "cpu",
        },
    )


def _request() -> RestartReplayRequestV1:
    return RestartReplayRequestV1(
        request_id="request-v2",
        simulator_configuration_id="configuration-v2",
        initial_state_id="endpoint-state-v1",
        group_log_scales=np.asarray([0.1, -0.2], dtype=np.float32),
        controller_points_m=np.zeros((4, 1, 3), dtype=np.float32),
        position_m=np.zeros((3, 3), dtype=np.float32),
        velocity_mps=np.ones((3, 3), dtype=np.float32),
        start_frame=2,
        stop_frame=4,
    )


def test_all_hit_rerun_skips_real_provider_construction(tmp_path: Path) -> None:
    counter = {"providers": 0, "replays": 0, "closes": 0}

    def factory(*args: Any, **kwargs: Any) -> FakeProvider:
        del args
        return FakeProvider(counter, **kwargs)

    args, kwargs = _factory_arguments()
    first_session = _session(tmp_path)
    first_proxy = first_session.wrap_factory(factory)(*args, **kwargs)
    assert isinstance(first_proxy, PhysTwinReplayProvider)
    first = first_proxy.replay(_request())
    first_proxy.close()

    second_session = _session(tmp_path)
    second_proxy = second_session.wrap_factory(factory)(*args, **kwargs)
    second = second_proxy.replay(_request())
    second_proxy.close()

    assert counter == {"providers": 1, "replays": 1, "closes": 1}
    np.testing.assert_array_equal(first.positions_m, second.positions_m)
    np.testing.assert_array_equal(first.velocities_mps, second.velocities_mps)
    np.testing.assert_array_equal(first.frame_ids, [2, 3])
    assert first_session.records[0]["cache_status"] == "miss"
    assert second_session.records[0]["cache_status"] == "hit"
    assert first_session.provider_instance_count == 1
    assert second_session.provider_instance_count == 0

    record = first_session.records[0]
    record_path = first_session.cache.root / record["record_path"]
    with np.load(record_path, allow_pickle=False) as archive:
        envelope = json.loads(str(archive["descriptor_json"]))
        assert "positions_m" in archive.files
        assert "velocities_mps" in archive.files
        assert str(archive["request_id"]) == "request-v2"
    descriptor = envelope["descriptor"]
    assert {
        "provider",
        "provider_source",
        "official_phystwin_source",
        "numerical_runtime",
        "source_artifacts_sha256",
        "spring_graph",
        "provider_factory",
        "request",
    } <= set(descriptor)
    assert descriptor["request"]["request_type"] == "RestartReplayRequestV1"
    assert record["positions_sha256"] != record["velocities_sha256"]
    assert first_session.manifest()["schema_version"] == 2
    assert first_session.manifest()["cached_payload"] == (
        "positions_velocities_and_provenance"
    )
