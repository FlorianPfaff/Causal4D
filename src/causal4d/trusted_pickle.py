"""Explicit, content-verified loading for trusted pickle inputs.

Pickle is executable code, not a data-only interchange format.  This module
therefore requires an explicit opt-in and verifies the exact bytes before
handing them to :mod:`pickle`.
"""

from __future__ import annotations

import hashlib
import pickle
import stat
from pathlib import Path
from typing import Any

_LOWER_HEX = frozenset("0123456789abcdef")


def _validated_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    return value


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"pickle path contains a symlink: {current}")


def load_trusted_pickle(
    path: str | Path,
    *,
    allow_unsafe_pickle: bool = False,
    expected_sha256: str | None = None,
) -> Any:
    """Load one explicitly trusted pickle after exact byte verification.

    ``allow_unsafe_pickle`` is deliberately false by default because unpickling
    can execute arbitrary code.  A digest establishes byte identity; it does
    not make an untrusted pickle safe.
    """

    if type(allow_unsafe_pickle) is not bool:
        raise TypeError("allow_unsafe_pickle must be an exact boolean")
    if not allow_unsafe_pickle:
        raise PermissionError(
            "pickle loading is disabled; pass allow_unsafe_pickle=True only "
            "for explicitly trusted, content-addressed inputs"
        )
    expected = _validated_sha256(expected_sha256)
    supplied = Path(path)
    _reject_symlink_components(supplied)
    try:
        metadata = supplied.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"trusted pickle does not exist: {supplied}") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"trusted pickle must be an ordinary file: {supplied}")

    payload = supplied.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(
            f"trusted pickle SHA-256 mismatch: {actual} != {expected}"
        )
    return pickle.loads(payload)


__all__ = ["load_trusted_pickle"]
