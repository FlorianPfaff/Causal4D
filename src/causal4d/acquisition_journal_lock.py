"""Fail-closed advisory locking for acquisition-journal mutation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib
from typing import BinaryIO, Protocol, cast


class _FcntlModule(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int, /) -> None: ...


try:
    _fcntl_module: object | None = importlib.import_module("fcntl")
except ModuleNotFoundError:  # pragma: no cover - fail-closed test replaces backend
    _fcntl_module = None

_fcntl = cast(_FcntlModule | None, _fcntl_module)
_LOCKING_ERROR = (
    "acquisition journal append and seal require POSIX advisory file locking"
)


def acquisition_journal_locking_available() -> bool:
    """Return whether journal mutation can obtain the required process lock."""

    return _fcntl is not None


def require_acquisition_journal_locking() -> _FcntlModule:
    """Return the locking backend or fail before any journal mutation occurs."""

    if _fcntl is None:
        raise RuntimeError(_LOCKING_ERROR)
    return _fcntl


@contextmanager
def exclusive_acquisition_journal_lock(handle: BinaryIO) -> Iterator[None]:
    """Hold an exclusive process lock for one append or seal transaction."""

    locking = require_acquisition_journal_locking()
    locking.flock(handle.fileno(), locking.LOCK_EX)
    try:
        yield
    finally:
        locking.flock(handle.fileno(), locking.LOCK_UN)


__all__ = [
    "acquisition_journal_locking_available",
    "exclusive_acquisition_journal_lock",
    "require_acquisition_journal_locking",
]
