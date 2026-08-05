"""Content-addressed, atomically published physical rollout records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Literal
import zipfile

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array

ROLLOUT_CACHE_SCHEMA_NAME = "causal4d.phystwin-rollout-cache"
ROLLOUT_CACHE_SCHEMA_VERSION = 1

CacheStatus = Literal["hit", "miss", "repaired", "race_hit"]

_SOURCE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cfg",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".ini",
        ".json",
        ".py",
        ".pyi",
        ".toml",
        ".yaml",
        ".yml",
    }
)
_SOURCE_FILENAMES = frozenset({"Dockerfile", "Makefile"})


class RolloutCacheValidationError(RuntimeError):
    """Raised when a cache record cannot satisfy its immutable contract."""


def _canonical_json(values: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(values),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("cache descriptors must contain finite JSON values") from error


def file_sha256(path: str | Path) -> str:
    """Hash one file without loading the complete payload into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_source_path(path: Path) -> bool:
    return path.name in _SOURCE_FILENAMES or path.suffix.lower() in _SOURCE_SUFFIXES


def _git_output(root: Path, *arguments: str, check: bool = True) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _untracked_git_files(root: Path) -> list[dict[str, str]]:
    output = _git_output(root, "ls-files", "--others", "--exclude-standard", "-z")
    entries: list[dict[str, str]] = []
    for raw_name in output.split(b"\0"):
        if not raw_name:
            continue
        relative = Path(os.fsdecode(raw_name))
        if not _is_source_path(relative):
            continue
        path = root / relative
        if path.is_symlink():
            digest = hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
            kind = "symlink"
        elif path.is_file():
            digest = file_sha256(path)
            kind = "file"
        else:
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "kind": kind,
                "sha256": digest,
            }
        )
    return sorted(entries, key=lambda value: value["path"])


def _content_tree_identity(root: Path) -> dict[str, Any]:
    skipped = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root)
        if any(part in skipped for part in relative.parts) or not _is_source_path(
            relative
        ):
            continue
        if path.is_symlink():
            entries.append(
                {
                    "path": relative.as_posix(),
                    "kind": "symlink",
                    "sha256": hashlib.sha256(
                        os.readlink(path).encode("utf-8")
                    ).hexdigest(),
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative.as_posix(),
                    "kind": "file",
                    "sha256": file_sha256(path),
                }
            )
    tree_sha256 = hashlib.sha256(
        _canonical_json({"entries": entries}).encode("utf-8")
    ).hexdigest()
    descriptor: dict[str, Any] = {
        "kind": "content_tree",
        "scope": "source_files_v1",
        "revision": "unversioned",
        "dirty": True,
        "file_count": len(entries),
        "tree_sha256": tree_sha256,
    }
    descriptor["fingerprint"] = hashlib.sha256(
        _canonical_json(descriptor).encode("utf-8")
    ).hexdigest()
    return descriptor


def repository_source_identity(path: str | Path) -> dict[str, Any]:
    """Identify a source checkout, including tracked and untracked modifications."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source repository does not exist: {root}")
    try:
        revision = _git_output(root, "rev-parse", "HEAD").decode("ascii").strip()
        diff = _git_output(root, "diff", "--binary", "HEAD", "--")
        untracked = _untracked_git_files(root)
        submodules = _git_output(
            root,
            "submodule",
            "status",
            "--recursive",
            check=False,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return _content_tree_identity(root)

    descriptor = {
        "kind": "git",
        "scope": "tracked_tree_and_untracked_source_files_v1",
        "revision": revision,
        "dirty": bool(
            diff
            or untracked
            or any(line[:1] in {b"+", b"-", b"U"} for line in submodules.splitlines())
        ),
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_source_files": untracked,
        "submodule_status_sha256": hashlib.sha256(submodules).hexdigest(),
    }
    descriptor["fingerprint"] = hashlib.sha256(
        _canonical_json(descriptor).encode("utf-8")
    ).hexdigest()
    return descriptor


def _validated_trajectory(
    values: np.ndarray,
    *,
    expected_frame_count: int,
    minimum_node_count: int,
) -> np.ndarray:
    trajectory = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if (
        trajectory.ndim != 3
        or trajectory.shape[0] != expected_frame_count
        or trajectory.shape[2] != 3
    ):
        raise ValueError("cached rollout must have shape (T, N, 3)")
    if trajectory.shape[1] < minimum_node_count:
        raise ValueError("cached rollout contains fewer nodes than requested")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("cached rollout must contain only finite values")
    return readonly_array(trajectory)


@dataclass(frozen=True)
class RolloutCacheResult:
    """One validated cache lookup or newly published immutable record."""

    trajectory: np.ndarray
    cache_key: str
    record_path: Path
    relative_path: str
    trajectory_sha256: str
    status: CacheStatus

    @property
    def cache_hit(self) -> bool:
        return self.status in {"hit", "race_hit"}


class ContentAddressedRolloutCache:
    """Store non-pickled rollout records below a SHA-256-addressed path."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def descriptor_json(descriptor: Mapping[str, Any]) -> str:
        envelope = {
            "schema_name": ROLLOUT_CACHE_SCHEMA_NAME,
            "schema_version": ROLLOUT_CACHE_SCHEMA_VERSION,
            "descriptor": dict(descriptor),
        }
        return _canonical_json(envelope)

    @classmethod
    def cache_key_for(cls, descriptor: Mapping[str, Any]) -> str:
        payload = cls.descriptor_json(descriptor).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _record_path(self, cache_key: str) -> Path:
        return self.root / cache_key[:2] / f"{cache_key}.npz"

    def _load_record(
        self,
        path: Path,
        *,
        cache_key: str,
        descriptor_json: str,
        expected_frame_count: int,
        minimum_node_count: int,
    ) -> tuple[np.ndarray, str]:
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "schema_name",
                    "schema_version",
                    "cache_key",
                    "descriptor_json",
                    "trajectory",
                    "trajectory_sha256",
                }
                missing = required - set(archive.files)
                if missing:
                    raise RolloutCacheValidationError(
                        "cache record is missing: " + ", ".join(sorted(missing))
                    )
                if str(np.asarray(archive["schema_name"]).item()) != (
                    ROLLOUT_CACHE_SCHEMA_NAME
                ):
                    raise RolloutCacheValidationError("cache schema name changed")
                if int(np.asarray(archive["schema_version"]).item()) != (
                    ROLLOUT_CACHE_SCHEMA_VERSION
                ):
                    raise RolloutCacheValidationError("cache schema version changed")
                if str(np.asarray(archive["cache_key"]).item()) != cache_key:
                    raise RolloutCacheValidationError(
                        "cache key does not match its path"
                    )
                stored_descriptor = str(np.asarray(archive["descriptor_json"]).item())
                if stored_descriptor != descriptor_json:
                    raise RolloutCacheValidationError(
                        "cache descriptor does not match the requested rollout"
                    )
                trajectory = _validated_trajectory(
                    archive["trajectory"],
                    expected_frame_count=expected_frame_count,
                    minimum_node_count=minimum_node_count,
                )
                stored_sha256 = str(np.asarray(archive["trajectory_sha256"]).item())
        except RolloutCacheValidationError:
            raise
        except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            raise RolloutCacheValidationError(
                f"cache record could not be decoded: {path}"
            ) from error

        actual_sha256 = array_sha256(trajectory)
        if stored_sha256 != actual_sha256:
            raise RolloutCacheValidationError("cached trajectory checksum mismatch")
        return trajectory, actual_sha256

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if not hasattr(os, "O_DIRECTORY"):
            return
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _publish_record(
        self,
        path: Path,
        *,
        cache_key: str,
        descriptor_json: str,
        trajectory: np.ndarray,
        trajectory_sha256: str,
        replace_existing: bool,
    ) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                prefix=f".{cache_key}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(
                    handle,
                    schema_name=np.asarray(ROLLOUT_CACHE_SCHEMA_NAME),
                    schema_version=np.asarray(ROLLOUT_CACHE_SCHEMA_VERSION),
                    cache_key=np.asarray(cache_key),
                    descriptor_json=np.asarray(descriptor_json),
                    trajectory=trajectory,
                    trajectory_sha256=np.asarray(trajectory_sha256),
                )
                handle.flush()
                os.fsync(handle.fileno())
            if replace_existing:
                os.replace(temporary_path, path)
                temporary_path = None
            else:
                try:
                    os.link(temporary_path, path)
                except FileExistsError:
                    return False
            self._fsync_directory(path.parent)
            return True
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def get_or_compute(
        self,
        descriptor: Mapping[str, Any],
        compute: Callable[[], np.ndarray],
        *,
        expected_frame_count: int,
        minimum_node_count: int,
    ) -> RolloutCacheResult:
        """Return a validated record, computing and publishing it only when absent."""

        if expected_frame_count < 1 or minimum_node_count < 1:
            raise ValueError("rollout cache dimensions must be positive")
        descriptor_json = self.descriptor_json(descriptor)
        cache_key = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
        path = self._record_path(cache_key)
        relative_path = path.relative_to(self.root).as_posix()
        invalidated = False

        if path.exists():
            try:
                trajectory, trajectory_sha256 = self._load_record(
                    path,
                    cache_key=cache_key,
                    descriptor_json=descriptor_json,
                    expected_frame_count=expected_frame_count,
                    minimum_node_count=minimum_node_count,
                )
            except RolloutCacheValidationError:
                invalidated = True
            else:
                return RolloutCacheResult(
                    trajectory=trajectory,
                    cache_key=cache_key,
                    record_path=path,
                    relative_path=relative_path,
                    trajectory_sha256=trajectory_sha256,
                    status="hit",
                )

        trajectory = _validated_trajectory(
            compute(),
            expected_frame_count=expected_frame_count,
            minimum_node_count=minimum_node_count,
        )
        trajectory_sha256 = array_sha256(trajectory)
        published = self._publish_record(
            path,
            cache_key=cache_key,
            descriptor_json=descriptor_json,
            trajectory=trajectory,
            trajectory_sha256=trajectory_sha256,
            replace_existing=invalidated,
        )
        try:
            loaded, loaded_sha256 = self._load_record(
                path,
                cache_key=cache_key,
                descriptor_json=descriptor_json,
                expected_frame_count=expected_frame_count,
                minimum_node_count=minimum_node_count,
            )
        except RolloutCacheValidationError:
            if published:
                raise
            self._publish_record(
                path,
                cache_key=cache_key,
                descriptor_json=descriptor_json,
                trajectory=trajectory,
                trajectory_sha256=trajectory_sha256,
                replace_existing=True,
            )
            loaded, loaded_sha256 = self._load_record(
                path,
                cache_key=cache_key,
                descriptor_json=descriptor_json,
                expected_frame_count=expected_frame_count,
                minimum_node_count=minimum_node_count,
            )
            status: CacheStatus = "repaired"
        else:
            if published:
                status = "repaired" if invalidated else "miss"
            else:
                status = "race_hit"
        return RolloutCacheResult(
            trajectory=loaded,
            cache_key=cache_key,
            record_path=path,
            relative_path=relative_path,
            trajectory_sha256=loaded_sha256,
            status=status,
        )


REPLAY_CACHE_SCHEMA_NAME = "causal4d.phystwin-replay-cache"
REPLAY_CACHE_SCHEMA_VERSION = 1


def _nonempty_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    identifier = value.strip()
    if not identifier:
        raise ValueError(f"{name} must be a nonempty identifier")
    return identifier


@dataclass(frozen=True)
class CachedReplayTrajectory:
    """Provider-neutral immutable representation of one replay-v2 response."""

    positions_m: np.ndarray
    velocities_mps: np.ndarray
    frame_ids: np.ndarray
    dt_s: float
    request_id: str
    simulator_configuration_id: str
    initial_state_id: str

    def __post_init__(self) -> None:
        raw_positions = np.asarray(self.positions_m)
        if raw_positions.ndim != 3:
            raise ValueError("cached replay positions must have shape (T, N, 3)")
        positions = _validated_trajectory(
            raw_positions,
            expected_frame_count=raw_positions.shape[0],
            minimum_node_count=1,
        )
        velocities = _validated_trajectory(
            self.velocities_mps,
            expected_frame_count=len(positions),
            minimum_node_count=positions.shape[1],
        )
        if velocities.shape != positions.shape:
            raise ValueError("cached replay positions and velocities must match")
        frame_ids = np.asarray(self.frame_ids, dtype=np.int64).copy()
        if frame_ids.shape != (len(positions),):
            raise ValueError("cached replay frame_ids must identify every frame")
        if np.any(frame_ids < 0) or (
            len(frame_ids) > 1 and np.any(np.diff(frame_ids) <= 0)
        ):
            raise ValueError(
                "cached replay frame_ids must be increasing and nonnegative"
            )
        frame_ids = readonly_integer_array(frame_ids, name="frame_ids")
        dt_s = float(self.dt_s)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("cached replay dt_s must be positive and finite")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "dt_s", dt_s)
        object.__setattr__(
            self,
            "request_id",
            _nonempty_identifier(self.request_id, name="request_id"),
        )
        object.__setattr__(
            self,
            "simulator_configuration_id",
            _nonempty_identifier(
                self.simulator_configuration_id,
                name="simulator_configuration_id",
            ),
        )
        object.__setattr__(
            self,
            "initial_state_id",
            _nonempty_identifier(self.initial_state_id, name="initial_state_id"),
        )

    @classmethod
    def from_object(cls, values: Any) -> CachedReplayTrajectory:
        """Copy a BPT ReplayTrajectoryV1 without importing the optional provider."""

        return cls(
            positions_m=np.asarray(values.positions_m),
            velocities_mps=np.asarray(values.velocities_mps),
            frame_ids=np.asarray(values.frame_ids),
            dt_s=float(values.dt_s),
            request_id=str(values.request_id),
            simulator_configuration_id=str(values.simulator_configuration_id),
            initial_state_id=str(values.initial_state_id),
        )


@dataclass(frozen=True)
class ReplayCacheResult:
    """One validated replay-v2 cache lookup or newly published record."""

    replay: CachedReplayTrajectory
    cache_key: str
    record_path: Path
    relative_path: str
    positions_sha256: str
    velocities_sha256: str
    status: CacheStatus

    @property
    def cache_hit(self) -> bool:
        return self.status in {"hit", "race_hit"}


class ContentAddressedReplayCache:
    """Atomically cache complete replay-v2 positions, velocities, and provenance."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def descriptor_json(descriptor: Mapping[str, Any]) -> str:
        return _canonical_json(
            {
                "schema_name": REPLAY_CACHE_SCHEMA_NAME,
                "schema_version": REPLAY_CACHE_SCHEMA_VERSION,
                "descriptor": dict(descriptor),
            }
        )

    @classmethod
    def cache_key_for(cls, descriptor: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            cls.descriptor_json(descriptor).encode("utf-8")
        ).hexdigest()

    def _record_path(self, cache_key: str) -> Path:
        return self.root / cache_key[:2] / f"{cache_key}.npz"

    @staticmethod
    def _validate_expected(
        replay: CachedReplayTrajectory,
        *,
        expected_frame_ids: np.ndarray,
        minimum_node_count: int,
        expected_dt_s: float,
        request_id: str,
        simulator_configuration_id: str,
        initial_state_id: str,
    ) -> None:
        frames = np.asarray(expected_frame_ids, dtype=np.int64)
        if replay.positions_m.shape[1] < minimum_node_count:
            raise ValueError("cached replay contains fewer nodes than requested")
        if not np.array_equal(replay.frame_ids, frames):
            raise ValueError("cached replay frame_ids do not match the request")
        if not np.isclose(replay.dt_s, expected_dt_s, rtol=0.0, atol=1e-15):
            raise ValueError("cached replay dt_s does not match the provider")
        if replay.request_id != request_id:
            raise ValueError("cached replay request_id does not match the request")
        if replay.simulator_configuration_id != simulator_configuration_id:
            raise ValueError("cached replay configuration identity changed")
        if replay.initial_state_id != initial_state_id:
            raise ValueError("cached replay initial-state identity changed")

    def _load_record(
        self,
        path: Path,
        *,
        cache_key: str,
        descriptor_json: str,
        expected_frame_ids: np.ndarray,
        minimum_node_count: int,
        expected_dt_s: float,
        request_id: str,
        simulator_configuration_id: str,
        initial_state_id: str,
    ) -> tuple[CachedReplayTrajectory, str, str]:
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "schema_name",
                    "schema_version",
                    "cache_key",
                    "descriptor_json",
                    "positions_m",
                    "velocities_mps",
                    "frame_ids",
                    "dt_s",
                    "request_id",
                    "simulator_configuration_id",
                    "initial_state_id",
                    "positions_sha256",
                    "velocities_sha256",
                }
                missing = required - set(archive.files)
                if missing:
                    raise RolloutCacheValidationError(
                        "replay cache record is missing: " + ", ".join(sorted(missing))
                    )
                if str(np.asarray(archive["schema_name"]).item()) != (
                    REPLAY_CACHE_SCHEMA_NAME
                ):
                    raise RolloutCacheValidationError(
                        "replay cache schema name changed"
                    )
                if int(np.asarray(archive["schema_version"]).item()) != (
                    REPLAY_CACHE_SCHEMA_VERSION
                ):
                    raise RolloutCacheValidationError(
                        "replay cache schema version changed"
                    )
                if str(np.asarray(archive["cache_key"]).item()) != cache_key:
                    raise RolloutCacheValidationError(
                        "replay cache key does not match its path"
                    )
                if (
                    str(np.asarray(archive["descriptor_json"]).item())
                    != descriptor_json
                ):
                    raise RolloutCacheValidationError(
                        "replay cache descriptor does not match the request"
                    )
                replay = CachedReplayTrajectory(
                    positions_m=archive["positions_m"],
                    velocities_mps=archive["velocities_mps"],
                    frame_ids=archive["frame_ids"],
                    dt_s=float(np.asarray(archive["dt_s"]).item()),
                    request_id=str(np.asarray(archive["request_id"]).item()),
                    simulator_configuration_id=str(
                        np.asarray(archive["simulator_configuration_id"]).item()
                    ),
                    initial_state_id=str(
                        np.asarray(archive["initial_state_id"]).item()
                    ),
                )
                stored_positions_sha256 = str(
                    np.asarray(archive["positions_sha256"]).item()
                )
                stored_velocities_sha256 = str(
                    np.asarray(archive["velocities_sha256"]).item()
                )
        except RolloutCacheValidationError:
            raise
        except (EOFError, OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            raise RolloutCacheValidationError(
                f"replay cache record could not be decoded: {path}"
            ) from error

        try:
            self._validate_expected(
                replay,
                expected_frame_ids=expected_frame_ids,
                minimum_node_count=minimum_node_count,
                expected_dt_s=expected_dt_s,
                request_id=request_id,
                simulator_configuration_id=simulator_configuration_id,
                initial_state_id=initial_state_id,
            )
        except ValueError as error:
            raise RolloutCacheValidationError(
                "cached replay provenance does not match the request"
            ) from error
        positions_sha256 = array_sha256(replay.positions_m)
        velocities_sha256 = array_sha256(replay.velocities_mps)
        if stored_positions_sha256 != positions_sha256:
            raise RolloutCacheValidationError(
                "cached replay position checksum mismatch"
            )
        if stored_velocities_sha256 != velocities_sha256:
            raise RolloutCacheValidationError(
                "cached replay velocity checksum mismatch"
            )
        return replay, positions_sha256, velocities_sha256

    def _publish_record(
        self,
        path: Path,
        *,
        cache_key: str,
        descriptor_json: str,
        replay: CachedReplayTrajectory,
        positions_sha256: str,
        velocities_sha256: str,
        replace_existing: bool,
    ) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=path.parent,
                prefix=f".{cache_key}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(
                    handle,
                    schema_name=np.asarray(REPLAY_CACHE_SCHEMA_NAME),
                    schema_version=np.asarray(REPLAY_CACHE_SCHEMA_VERSION),
                    cache_key=np.asarray(cache_key),
                    descriptor_json=np.asarray(descriptor_json),
                    positions_m=replay.positions_m,
                    velocities_mps=replay.velocities_mps,
                    frame_ids=replay.frame_ids,
                    dt_s=np.asarray(replay.dt_s),
                    request_id=np.asarray(replay.request_id),
                    simulator_configuration_id=np.asarray(
                        replay.simulator_configuration_id
                    ),
                    initial_state_id=np.asarray(replay.initial_state_id),
                    positions_sha256=np.asarray(positions_sha256),
                    velocities_sha256=np.asarray(velocities_sha256),
                )
                handle.flush()
                os.fsync(handle.fileno())
            if replace_existing:
                os.replace(temporary_path, path)
                temporary_path = None
            else:
                try:
                    os.link(temporary_path, path)
                except FileExistsError:
                    return False
            ContentAddressedRolloutCache._fsync_directory(path.parent)
            return True
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def get_or_compute(
        self,
        descriptor: Mapping[str, Any],
        compute: Callable[[], Any],
        *,
        expected_frame_ids: np.ndarray,
        minimum_node_count: int,
        expected_dt_s: float,
        request_id: str,
        simulator_configuration_id: str,
        initial_state_id: str,
    ) -> ReplayCacheResult:
        """Return a complete validated replay, computing it only when absent."""

        frames = np.asarray(expected_frame_ids, dtype=np.int64)
        if frames.ndim != 1 or not len(frames) or minimum_node_count < 1:
            raise ValueError("replay cache dimensions must be positive")
        expected_dt = float(expected_dt_s)
        if not np.isfinite(expected_dt) or expected_dt <= 0.0:
            raise ValueError("expected replay dt_s must be positive and finite")
        expected_request_id = _nonempty_identifier(request_id, name="request_id")
        expected_configuration_id = _nonempty_identifier(
            simulator_configuration_id,
            name="simulator_configuration_id",
        )
        expected_initial_state_id = _nonempty_identifier(
            initial_state_id,
            name="initial_state_id",
        )
        descriptor_json = self.descriptor_json(descriptor)
        cache_key = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
        path = self._record_path(cache_key)
        relative_path = path.relative_to(self.root).as_posix()
        invalidated = False

        if path.exists():
            try:
                replay, positions_sha256, velocities_sha256 = self._load_record(
                    path,
                    cache_key=cache_key,
                    descriptor_json=descriptor_json,
                    expected_frame_ids=frames,
                    minimum_node_count=minimum_node_count,
                    expected_dt_s=expected_dt,
                    request_id=expected_request_id,
                    simulator_configuration_id=expected_configuration_id,
                    initial_state_id=expected_initial_state_id,
                )
            except RolloutCacheValidationError:
                invalidated = True
            else:
                return ReplayCacheResult(
                    replay=replay,
                    cache_key=cache_key,
                    record_path=path,
                    relative_path=relative_path,
                    positions_sha256=positions_sha256,
                    velocities_sha256=velocities_sha256,
                    status="hit",
                )

        replay = CachedReplayTrajectory.from_object(compute())
        self._validate_expected(
            replay,
            expected_frame_ids=frames,
            minimum_node_count=minimum_node_count,
            expected_dt_s=expected_dt,
            request_id=expected_request_id,
            simulator_configuration_id=expected_configuration_id,
            initial_state_id=expected_initial_state_id,
        )
        positions_sha256 = array_sha256(replay.positions_m)
        velocities_sha256 = array_sha256(replay.velocities_mps)
        published = self._publish_record(
            path,
            cache_key=cache_key,
            descriptor_json=descriptor_json,
            replay=replay,
            positions_sha256=positions_sha256,
            velocities_sha256=velocities_sha256,
            replace_existing=invalidated,
        )
        try:
            loaded, loaded_positions_sha256, loaded_velocities_sha256 = (
                self._load_record(
                    path,
                    cache_key=cache_key,
                    descriptor_json=descriptor_json,
                    expected_frame_ids=frames,
                    minimum_node_count=minimum_node_count,
                    expected_dt_s=expected_dt,
                    request_id=expected_request_id,
                    simulator_configuration_id=expected_configuration_id,
                    initial_state_id=expected_initial_state_id,
                )
            )
        except RolloutCacheValidationError:
            if published:
                raise
            self._publish_record(
                path,
                cache_key=cache_key,
                descriptor_json=descriptor_json,
                replay=replay,
                positions_sha256=positions_sha256,
                velocities_sha256=velocities_sha256,
                replace_existing=True,
            )
            loaded, loaded_positions_sha256, loaded_velocities_sha256 = (
                self._load_record(
                    path,
                    cache_key=cache_key,
                    descriptor_json=descriptor_json,
                    expected_frame_ids=frames,
                    minimum_node_count=minimum_node_count,
                    expected_dt_s=expected_dt,
                    request_id=expected_request_id,
                    simulator_configuration_id=expected_configuration_id,
                    initial_state_id=expected_initial_state_id,
                )
            )
            status: CacheStatus = "repaired"
        else:
            if published:
                status = "repaired" if invalidated else "miss"
            else:
                status = "race_hit"
        return ReplayCacheResult(
            replay=loaded,
            cache_key=cache_key,
            record_path=path,
            relative_path=relative_path,
            positions_sha256=loaded_positions_sha256,
            velocities_sha256=loaded_velocities_sha256,
            status=status,
        )
