"""Content-addressed, fail-closed cache for expensive PhysTwin restart rollouts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.contracts import array_sha256

PHYSTWIN_ROLLOUT_CACHE_SCHEMA = "causal4d.phystwin_rollout_cache"
PHYSTWIN_ROLLOUT_CACHE_VERSION = 1


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("cache metadata must contain finite JSON values") from error


def _is_digest(value: object, *, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _require_digest(value: object, *, name: str, length: int = 64) -> str:
    if not _is_digest(value, length=length):
        raise ValueError(f"{name} must be a lowercase {length}-hex digest")
    return str(value)


def file_sha256(path: str | Path) -> str:
    """Hash a source artifact without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_clean_git_revision(repository: str | Path) -> str:
    """Return an exact revision and reject dirty or non-Git upstream checkouts."""

    root = Path(repository).resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "rollout caching requires a readable official PhysTwin Git checkout"
        ) from error
    _require_digest(revision, name="official PhysTwin revision", length=40)
    if status.strip():
        raise ValueError(
            "rollout caching requires a clean official PhysTwin checkout"
        )
    return revision


@dataclass(frozen=True)
class PhysTwinRolloutCacheKeyV1:
    """Complete deterministic identity of one provider restart rollout."""

    replay_provider_manifest_id: str
    graph_provider_manifest_id: str
    official_phystwin_revision: str
    source_artifact_sha256: Mapping[str, str]
    graph_vertices_sha256: str
    graph_springs_sha256: str
    graph_rest_lengths_sha256: str
    graph_masses_sha256: str
    graph_num_object_springs: int
    graph_num_object_points: int
    controller_points_sha256: str
    endpoint_position_sha256: str
    endpoint_velocity_sha256: str
    group_log_scales_sha256: str
    start_frame: int
    stop_frame: int
    runtime: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_digest(
            self.replay_provider_manifest_id,
            name="replay provider manifest id",
        )
        _require_digest(
            self.graph_provider_manifest_id,
            name="graph provider manifest id",
        )
        _require_digest(
            self.official_phystwin_revision,
            name="official PhysTwin revision",
            length=40,
        )
        sources = {str(name): str(value) for name, value in self.source_artifact_sha256.items()}
        if not sources or any(not name for name in sources):
            raise ValueError("source artifact digests must be named and nonempty")
        for name, value in sources.items():
            _require_digest(value, name=f"source artifact {name}")
        for name, value in (
            ("graph vertices", self.graph_vertices_sha256),
            ("graph springs", self.graph_springs_sha256),
            ("graph rest lengths", self.graph_rest_lengths_sha256),
            ("graph masses", self.graph_masses_sha256),
            ("controller points", self.controller_points_sha256),
            ("endpoint position", self.endpoint_position_sha256),
            ("endpoint velocity", self.endpoint_velocity_sha256),
            ("group log scales", self.group_log_scales_sha256),
        ):
            _require_digest(value, name=name)
        if self.graph_num_object_springs < 0 or self.graph_num_object_points < 1:
            raise ValueError("graph counts are invalid")
        if not 0 <= self.start_frame < self.stop_frame:
            raise ValueError("rollout frame interval must be nonempty")
        runtime = json.loads(_canonical_json(self.runtime))
        object.__setattr__(self, "source_artifact_sha256", dict(sorted(sources.items())))
        object.__setattr__(self, "runtime", runtime)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_name": PHYSTWIN_ROLLOUT_CACHE_SCHEMA,
            "schema_version": PHYSTWIN_ROLLOUT_CACHE_VERSION,
            "replay_provider_manifest_id": self.replay_provider_manifest_id,
            "graph_provider_manifest_id": self.graph_provider_manifest_id,
            "official_phystwin_revision": self.official_phystwin_revision,
            "source_artifact_sha256": dict(self.source_artifact_sha256),
            "graph": {
                "vertices_sha256": self.graph_vertices_sha256,
                "springs_sha256": self.graph_springs_sha256,
                "rest_lengths_sha256": self.graph_rest_lengths_sha256,
                "masses_sha256": self.graph_masses_sha256,
                "num_object_springs": self.graph_num_object_springs,
                "num_object_points": self.graph_num_object_points,
            },
            "controller_points_sha256": self.controller_points_sha256,
            "endpoint_position_sha256": self.endpoint_position_sha256,
            "endpoint_velocity_sha256": self.endpoint_velocity_sha256,
            "group_log_scales_sha256": self.group_log_scales_sha256,
            "start_frame": self.start_frame,
            "stop_frame": self.stop_frame,
            "runtime": dict(self.runtime),
        }

    @property
    def cache_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.descriptor())).hexdigest()


def build_phystwin_rollout_cache_key(
    *,
    replay_provider_manifest_id: str,
    graph_provider_manifest_id: str,
    official_phystwin_revision: str,
    source_artifact_sha256: Mapping[str, str],
    graph_vertices: np.ndarray,
    graph_springs: np.ndarray,
    graph_rest_lengths: np.ndarray,
    graph_masses: np.ndarray,
    graph_num_object_springs: int,
    graph_num_object_points: int,
    controller_points: np.ndarray,
    endpoint_position: np.ndarray,
    endpoint_velocity: np.ndarray,
    group_log_scales: np.ndarray,
    start_frame: int,
    stop_frame: int,
    runtime: Mapping[str, Any],
) -> PhysTwinRolloutCacheKeyV1:
    """Build a cache key from every input that can alter a restart rollout."""

    return PhysTwinRolloutCacheKeyV1(
        replay_provider_manifest_id=replay_provider_manifest_id,
        graph_provider_manifest_id=graph_provider_manifest_id,
        official_phystwin_revision=official_phystwin_revision,
        source_artifact_sha256=source_artifact_sha256,
        graph_vertices_sha256=array_sha256(graph_vertices),
        graph_springs_sha256=array_sha256(graph_springs),
        graph_rest_lengths_sha256=array_sha256(graph_rest_lengths),
        graph_masses_sha256=array_sha256(graph_masses),
        graph_num_object_springs=int(graph_num_object_springs),
        graph_num_object_points=int(graph_num_object_points),
        controller_points_sha256=array_sha256(controller_points),
        endpoint_position_sha256=array_sha256(endpoint_position),
        endpoint_velocity_sha256=array_sha256(endpoint_velocity),
        group_log_scales_sha256=array_sha256(group_log_scales),
        start_frame=int(start_frame),
        stop_frame=int(stop_frame),
        runtime=runtime,
    )


@dataclass(frozen=True)
class PhysTwinRolloutCache:
    """Atomic directory-backed store for validated restart trajectories."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    def path_for(self, key: PhysTwinRolloutCacheKeyV1) -> Path:
        return self.root / key.cache_id[:2] / f"{key.cache_id}.npz"

    def load(
        self,
        key: PhysTwinRolloutCacheKeyV1,
        *,
        expected_shape: Sequence[int] | None = None,
    ) -> np.ndarray | None:
        """Load and verify one cache entry, returning ``None`` only when absent."""

        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "cache_id",
                    "descriptor_json",
                    "trajectory",
                    "trajectory_sha256",
                }
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(f"rollout cache entry is missing {sorted(missing)}")
                cache_id = str(archive["cache_id"].item())
                descriptor = json.loads(str(archive["descriptor_json"].item()))
                trajectory = np.asarray(archive["trajectory"]).copy()
                stored_sha256 = str(archive["trajectory_sha256"].item())
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid rollout cache entry: {path}") from error
        if cache_id != key.cache_id or descriptor != key.descriptor():
            raise ValueError("rollout cache descriptor does not match its lookup key")
        _require_digest(stored_sha256, name="cached trajectory")
        if array_sha256(trajectory) != stored_sha256:
            raise ValueError("cached trajectory digest does not match its payload")
        if expected_shape is not None and trajectory.shape != tuple(expected_shape):
            raise ValueError(
                "cached trajectory shape differs from the requested rollout shape"
            )
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("cached trajectory contains nonfinite values")
        trajectory.setflags(write=False)
        return trajectory

    def store(
        self,
        key: PhysTwinRolloutCacheKeyV1,
        trajectory: np.ndarray,
        *,
        expected_shape: Sequence[int] | None = None,
    ) -> Path:
        """Atomically persist one verified trajectory without overwriting conflicts."""

        values = np.asarray(trajectory)
        if expected_shape is not None and values.shape != tuple(expected_shape):
            raise ValueError("rollout trajectory has an unexpected shape")
        if values.ndim != 3 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("rollout trajectory must have finite shape (T, N, 3)")
        existing = self.load(key, expected_shape=expected_shape)
        if existing is not None:
            if existing.dtype != values.dtype or not np.array_equal(existing, values):
                raise ValueError("content-addressed rollout cache collision")
            return self.path_for(key)

        target = self.path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor_json = json.dumps(
            key.descriptor(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        trajectory_sha256 = array_sha256(values)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{key.cache_id}.",
                suffix=".npz",
                dir=target.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            np.savez_compressed(
                temporary,
                cache_id=np.asarray(key.cache_id),
                descriptor_json=np.asarray(descriptor_json),
                trajectory=values,
                trajectory_sha256=np.asarray(trajectory_sha256),
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        verified = self.load(key, expected_shape=expected_shape)
        if verified is None or verified.dtype != values.dtype or not np.array_equal(
            verified, values
        ):
            raise ValueError("new rollout cache entry did not verify after writing")
        return target


__all__ = [
    "PHYSTWIN_ROLLOUT_CACHE_SCHEMA",
    "PHYSTWIN_ROLLOUT_CACHE_VERSION",
    "PhysTwinRolloutCache",
    "PhysTwinRolloutCacheKeyV1",
    "build_phystwin_rollout_cache_key",
    "file_sha256",
    "require_clean_git_revision",
]
