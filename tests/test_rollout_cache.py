from pathlib import Path

import numpy as np
import pytest

from causal4d.rollout_cache import (
    CachedReplayTrajectory,
    ContentAddressedReplayCache,
    ContentAddressedRolloutCache,
    repository_source_identity,
)


def test_cache_reuses_validated_record_without_recomputing(tmp_path: Path) -> None:
    cache = ContentAddressedRolloutCache(tmp_path / "cache")
    descriptor = {"case": "unit", "dt": 0.1}
    calls = 0

    def compute() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.arange(18, dtype=np.float32).reshape(2, 3, 3)

    first = cache.get_or_compute(
        descriptor,
        compute,
        expected_frame_count=2,
        minimum_node_count=3,
    )
    second = cache.get_or_compute(
        descriptor,
        compute,
        expected_frame_count=2,
        minimum_node_count=3,
    )

    assert calls == 1
    assert first.status == "miss"
    assert second.status == "hit"
    assert first.cache_key == second.cache_key
    assert np.array_equal(first.trajectory, second.trajectory)
    assert not second.trajectory.flags.writeable


def test_cache_repairs_corrupt_record(tmp_path: Path) -> None:
    cache = ContentAddressedRolloutCache(tmp_path / "cache")
    descriptor = {"case": "unit"}
    calls = 0

    def compute() -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones((2, 2, 3), dtype=np.float32) * calls

    first = cache.get_or_compute(
        descriptor,
        compute,
        expected_frame_count=2,
        minimum_node_count=2,
    )
    first.record_path.write_bytes(b"not an npz")
    repaired = cache.get_or_compute(
        descriptor,
        compute,
        expected_frame_count=2,
        minimum_node_count=2,
    )

    assert calls == 2
    assert repaired.status == "repaired"
    assert np.all(repaired.trajectory == 2.0)


def test_cache_key_changes_with_physical_input() -> None:
    first = ContentAddressedRolloutCache.cache_key_for(
        {"endpoint": {"sha256": "a" * 64}, "dt": 1e-4}
    )
    second = ContentAddressedRolloutCache.cache_key_for(
        {"endpoint": {"sha256": "b" * 64}, "dt": 1e-4}
    )
    assert first != second


def test_repository_identity_changes_with_unversioned_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "simulator.py"
    path.write_text("value = 1\n", encoding="utf-8")
    first = repository_source_identity(source)
    path.write_text("value = 2\n", encoding="utf-8")
    second = repository_source_identity(source)

    assert first["kind"] == "content_tree"
    assert first["fingerprint"] != second["fingerprint"]


def test_invalid_computed_trajectory_is_not_published(tmp_path: Path) -> None:
    cache = ContentAddressedRolloutCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="shape"):
        cache.get_or_compute(
            {"case": "unit"},
            lambda: np.zeros((2, 3), dtype=np.float32),
            expected_frame_count=2,
            minimum_node_count=1,
        )
    assert not tuple((tmp_path / "cache").rglob("*.npz"))


def _cached_replay(value: float = 1.0) -> CachedReplayTrajectory:
    return CachedReplayTrajectory(
        positions_m=np.full((2, 3, 3), value, dtype=np.float32),
        velocities_mps=np.full((2, 3, 3), value + 1.0, dtype=np.float32),
        frame_ids=np.asarray([4, 5]),
        dt_s=0.03,
        request_id="request-v2",
        simulator_configuration_id="configuration-v2",
        initial_state_id="endpoint-state-v1",
    )


def test_replay_cache_reuses_positions_velocities_and_provenance(
    tmp_path: Path,
) -> None:
    cache = ContentAddressedReplayCache(tmp_path / "replay-cache")
    calls = 0

    def compute() -> CachedReplayTrajectory:
        nonlocal calls
        calls += 1
        return _cached_replay()

    kwargs = {
        "expected_frame_ids": np.asarray([4, 5]),
        "minimum_node_count": 3,
        "expected_dt_s": 0.03,
        "request_id": "request-v2",
        "simulator_configuration_id": "configuration-v2",
        "initial_state_id": "endpoint-state-v1",
    }
    first = cache.get_or_compute({"case": "unit"}, compute, **kwargs)
    second = cache.get_or_compute({"case": "unit"}, compute, **kwargs)

    assert calls == 1
    assert first.status == "miss"
    assert second.status == "hit"
    np.testing.assert_array_equal(first.replay.positions_m, second.replay.positions_m)
    np.testing.assert_array_equal(
        first.replay.velocities_mps,
        second.replay.velocities_mps,
    )
    assert first.positions_sha256 != first.velocities_sha256
    assert not second.replay.positions_m.flags.writeable
    assert not second.replay.velocities_mps.flags.writeable


def test_replay_cache_rejects_mismatched_response_without_publishing(
    tmp_path: Path,
) -> None:
    cache = ContentAddressedReplayCache(tmp_path / "replay-cache")
    with pytest.raises(ValueError, match="request_id"):
        cache.get_or_compute(
            {"case": "unit"},
            lambda: CachedReplayTrajectory(
                **{
                    **_cached_replay().__dict__,
                    "request_id": "wrong-request",
                }
            ),
            expected_frame_ids=np.asarray([4, 5]),
            minimum_node_count=3,
            expected_dt_s=0.03,
            request_id="request-v2",
            simulator_configuration_id="configuration-v2",
            initial_state_id="endpoint-state-v1",
        )
    assert not tuple((tmp_path / "replay-cache").rglob("*.npz"))

def test_replay_cache_repairs_semantically_tampered_record(tmp_path: Path) -> None:
    cache = ContentAddressedReplayCache(tmp_path / "replay-cache")
    calls = 0

    def compute() -> CachedReplayTrajectory:
        nonlocal calls
        calls += 1
        return _cached_replay(float(calls))

    kwargs = {
        "expected_frame_ids": np.asarray([4, 5]),
        "minimum_node_count": 3,
        "expected_dt_s": 0.03,
        "request_id": "request-v2",
        "simulator_configuration_id": "configuration-v2",
        "initial_state_id": "endpoint-state-v1",
    }
    first = cache.get_or_compute({"case": "unit"}, compute, **kwargs)
    with np.load(first.record_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["frame_ids"] = np.asarray([3, 5])
    with first.record_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)

    repaired = cache.get_or_compute({"case": "unit"}, compute, **kwargs)

    assert calls == 2
    assert repaired.status == "repaired"
    np.testing.assert_array_equal(repaired.replay.frame_ids, [4, 5])
    assert np.all(repaired.replay.positions_m == 2.0)


def test_cached_replay_rejects_nontrajectory_shapes_and_identifiers() -> None:
    values = _cached_replay().__dict__
    with pytest.raises(ValueError, match="positions"):
        CachedReplayTrajectory(**{**values, "positions_m": np.zeros(3)})
    with pytest.raises(TypeError, match="request_id"):
        CachedReplayTrajectory(**{**values, "request_id": None})
