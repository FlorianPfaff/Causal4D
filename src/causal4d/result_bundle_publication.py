"""Atomic whole-directory publication for flat Causal4D result bundles."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json
from causal4d.result_bundle_verification import verify_embedded_result_bundle

ResultBundleWriter = Callable[[Path], None]


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    if os.name != "posix":
        return
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"publication path contains a symlink: {current}")


def _artifact_inventory(staging: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for entry in sorted(staging.iterdir(), key=lambda path: path.name):
        if entry.name == "manifest.json":
            raise ValueError("bundle writer must not create manifest.json")
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                "result bundle writers may create only flat ordinary files: "
                f"{entry.name}"
            )
        _fsync_file(entry)
        artifacts[entry.name] = {
            "bytes": entry.stat().st_size,
            "sha256": _sha256(entry),
        }
    if not artifacts:
        raise ValueError("result bundle must contain at least one artifact")
    return artifacts


def _acquire_lock(lock_path: Path) -> None:
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            f"result bundle publication is already locked: {lock_path}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(lock_path.parent)


def publish_result_bundle(
    target_directory: str | Path,
    *,
    benchmark: str,
    writer: ResultBundleWriter,
) -> dict[str, Any]:
    """Build, verify, and atomically expose one immutable result directory.

    The final destination is exactly-once: an existing target is never replaced.
    The writer receives a same-parent ``*.incomplete`` directory.  Only after
    every artifact is flushed, a content manifest is written, and the complete
    staging bundle verifies is the directory renamed to its final name.
    """

    if type(benchmark) is not str or not benchmark:
        raise ValueError("benchmark must be a nonempty exact string")
    if not callable(writer):
        raise TypeError("writer must be callable")

    target = Path(target_directory)
    if not target.name or target.name in {".", ".."}:
        raise ValueError("target_directory must name a result bundle")
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent)
    if target.is_symlink():
        raise ValueError("target_directory must not be a symlink")

    lock_path = target.parent / f".{target.name}.publish.lock"
    _acquire_lock(lock_path)
    staging: Path | None = None
    try:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"result bundle already exists: {target}")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.",
                suffix=".incomplete",
                dir=target.parent,
            )
        )
        writer(staging)
        if not staging.is_dir() or staging.is_symlink():
            raise ValueError("bundle writer replaced the staging directory")

        artifacts = _artifact_inventory(staging)
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "benchmark": benchmark,
                "artifacts": artifacts,
            },
            overwrite=False,
        )
        _fsync_file(staging / "manifest.json")
        _fsync_directory(staging)

        staged_verification = verify_embedded_result_bundle(staging)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"result bundle appeared during publication: {target}"
            )
        os.rename(staging, target)
        staging = None
        _fsync_directory(target.parent)

        final_verification = verify_embedded_result_bundle(target)
        if (
            final_verification["manifest_sha256"]
            != staged_verification["manifest_sha256"]
            or final_verification["artifacts"] != staged_verification["artifacts"]
        ):
            raise RuntimeError("published result bundle changed during rename")
        return final_verification
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        lock_path.unlink(missing_ok=True)
        _fsync_directory(target.parent)


__all__ = ["ResultBundleWriter", "publish_result_bundle"]
