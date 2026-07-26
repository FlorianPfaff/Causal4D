from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from causal4d.phystwin_rollout_cache import (
    PhysTwinRolloutCache,
    build_phystwin_rollout_cache_key,
    file_sha256,
    require_clean_git_revision,
)


def _key(*, controller_offset: float = 0.0, substeps: int = 8):
    return build_phystwin_rollout_cache_key(
        replay_provider_manifest_id="1" * 64,
        graph_provider_manifest_id="2" * 64,
        official_phystwin_revision="3" * 40,
        source_artifact_sha256={
            "checkpoint": "4" * 64,
            "final_data": "5" * 64,
            "optimal_params": "6" * 64,
            "parameter_profile": "7" * 64,
        },
        graph_vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        graph_springs=np.asarray([[0, 1]], dtype=np.int32),
        graph_rest_lengths=np.asarray([1.0], dtype=np.float32),
        graph_masses=np.asarray([1.0, 1.0], dtype=np.float32),
        graph_num_object_springs=1,
        graph_num_object_points=2,
        controller_points=np.asarray(
            [[[controller_offset, 0.0, 0.0]]], dtype=np.float32
        ),
        endpoint_position=np.zeros((2, 3), dtype=np.float32),
        endpoint_velocity=np.zeros((2, 3), dtype=np.float32),
        group_log_scales=np.asarray([0.1, -0.2], dtype=np.float32),
        start_frame=4,
        stop_frame=7,
        runtime={
            "deterministic_spring_forces": True,
            "device": "cuda:0",
            "dt": 5e-5,
            "num_substeps": substeps,
            "self_collision": False,
            "spring_parameterization": "grouped",
        },
    )


def test_cache_key_is_stable_and_covers_rollout_inputs() -> None:
    first = _key()
    second = _key()
    assert first.cache_id == second.cache_id
    assert first.descriptor() == second.descriptor()
    assert _key(controller_offset=0.1).cache_id != first.cache_id
    assert _key(substeps=9).cache_id != first.cache_id


def test_cache_round_trip_is_atomic_validated_and_read_only(tmp_path: Path) -> None:
    cache = PhysTwinRolloutCache(tmp_path / "cache")
    key = _key()
    values = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    assert cache.load(key, expected_shape=values.shape) is None

    path = cache.store(key, values, expected_shape=values.shape)
    assert path == cache.path_for(key)
    assert path.is_file()
    loaded = cache.load(key, expected_shape=values.shape)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, values)
    assert loaded.dtype == values.dtype
    assert not loaded.flags.writeable

    assert cache.store(key, values.copy(), expected_shape=values.shape) == path
    with pytest.raises(ValueError, match="collision"):
        cache.store(key, values + 1.0, expected_shape=values.shape)


def test_cache_rejects_tampering_and_shape_drift(tmp_path: Path) -> None:
    cache = PhysTwinRolloutCache(tmp_path)
    key = _key()
    values = np.zeros((3, 2, 3), dtype=np.float32)
    path = cache.store(key, values, expected_shape=values.shape)

    with pytest.raises(ValueError, match="shape"):
        cache.load(key, expected_shape=(4, 2, 3))

    np.savez_compressed(
        path,
        cache_id=np.asarray(key.cache_id),
        descriptor_json=np.asarray(
            json.dumps(key.descriptor(), sort_keys=True, separators=(",", ":"))
        ),
        trajectory=np.ones_like(values),
        trajectory_sha256=np.asarray("0" * 64),
    )
    with pytest.raises(ValueError, match="digest"):
        cache.load(key, expected_shape=values.shape)


def test_cache_validates_source_files_and_upstream_git_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"causal4d")
    assert file_sha256(source) == hashlib.sha256(b"causal4d").hexdigest()

    revision = "a" * 40

    def clean_run(command, **kwargs):
        del kwargs
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=revision + "\n")
        if command[-3:] == ["--porcelain=v1", "--untracked-files=all"]:
            return SimpleNamespace(stdout="")
        raise AssertionError(command)

    monkeypatch.setattr("causal4d.phystwin_rollout_cache.subprocess.run", clean_run)
    assert require_clean_git_revision(tmp_path) == revision

    def dirty_run(command, **kwargs):
        del kwargs
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=revision + "\n")
        return SimpleNamespace(stdout=" M simulator.py\n")

    monkeypatch.setattr("causal4d.phystwin_rollout_cache.subprocess.run", dirty_run)
    with pytest.raises(ValueError, match="clean"):
        require_clean_git_revision(tmp_path)
