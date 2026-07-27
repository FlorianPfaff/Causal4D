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
    trajectory.setflags(write=False)
    return trajectory


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
