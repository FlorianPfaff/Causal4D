import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from causal4d.phystwin_resumable import _RolloutCacheSession
from causal4d.rollout_cache import ContentAddressedRolloutCache


class FakeProvider:
    def __init__(self, counter: dict[str, int]) -> None:
        counter["providers"] += 1
        self.counter = counter
        self.controller: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.device = "cpu"

    def set_controller_points(self, values: np.ndarray) -> None:
        self.controller = np.asarray(values).copy()

    def set_group_log_scales(self, values: np.ndarray) -> None:
        self.scales = np.asarray(values).copy()

    def replay_initial(self, *, frame_count: int) -> tuple[np.ndarray, None]:
        return np.zeros((frame_count, 1, 3), dtype=np.float32), None

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        del velocity_mps
        self.counter["replays"] += 1
        frames = stop_frame - start_frame
        endpoint = np.asarray(position_m, dtype=np.float32)
        result = np.repeat(endpoint[None], frames, axis=0)
        if self.scales is None:
            raise RuntimeError("scales were not set")
        result += float(np.sum(self.scales))
        return result

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
        cache=ContentAddressedRolloutCache(tmp_path / "cache"),
        provider_manifest={
            "manifest_id": "provider",
            "provider_version": "0.4.1",
            "provider_revision": "revision",
            "schema_version": 1,
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
            "deterministic_spring_forces": True,
            "spring_parameterization": "grouped",
            "device": "cpu",
        },
    )


def _replay(proxy: Any) -> np.ndarray:
    proxy.set_controller_points(np.zeros((4, 1, 3), dtype=np.float32))
    proxy.set_group_log_scales(np.asarray([0.1, -0.2], dtype=np.float32))
    return proxy.replay_restart(
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        start_frame=2,
        stop_frame=4,
    )


def test_all_hit_rerun_skips_real_provider_construction(tmp_path: Path) -> None:
    counter = {"providers": 0, "replays": 0, "closes": 0}

    def factory(*args: Any, **kwargs: Any) -> FakeProvider:
        del args, kwargs
        return FakeProvider(counter)

    args, kwargs = _factory_arguments()
    first_session = _session(tmp_path)
    first_proxy = first_session.wrap_factory(factory)(*args, **kwargs)
    first = _replay(first_proxy)
    first_proxy.close()

    second_session = _session(tmp_path)
    second_proxy = second_session.wrap_factory(factory)(*args, **kwargs)
    second = _replay(second_proxy)
    second_proxy.close()

    assert counter == {"providers": 1, "replays": 1, "closes": 1}
    assert np.array_equal(first, second)
    assert first_session.records[0]["cache_status"] == "miss"
    assert second_session.records[0]["cache_status"] == "hit"
    assert first_session.provider_instance_count == 1
    assert second_session.provider_instance_count == 0

    record = first_session.records[0]
    record_path = first_session.cache.root / record["record_path"]
    with np.load(record_path, allow_pickle=False) as archive:
        envelope = json.loads(str(archive["descriptor_json"]))
    descriptor = envelope["descriptor"]
    assert {
        "provider",
        "provider_source",
        "official_phystwin_source",
        "numerical_runtime",
        "source_artifacts_sha256",
        "spring_graph",
        "controller_points",
        "group_log_scales",
        "endpoint_position",
        "endpoint_velocity",
        "frame_interval",
    } <= set(descriptor)
