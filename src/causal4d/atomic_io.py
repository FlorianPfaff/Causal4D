"""Atomic publication helpers for finite JSON evidence artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry update on platforms that support it."""

    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = True,
) -> None:
    """Publish text atomically without leaving a partial destination file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
            temporary.unlink()
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    overwrite: bool = True,
    indent: int | None = 2,
) -> None:
    """Serialize finite, key-sorted JSON and publish it atomically."""

    serialized = json.dumps(
        payload,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(
        path,
        serialized + "\n",
        overwrite=overwrite,
    )


__all__ = ["atomic_write_json", "atomic_write_text"]
