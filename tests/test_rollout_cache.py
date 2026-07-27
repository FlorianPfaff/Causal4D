from pathlib import Path

import numpy as np
import pytest

from causal4d.rollout_cache import (
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
