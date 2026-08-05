"""Owned, irreversibly read-only NumPy array helpers for frozen artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np


def _immutable_buffer_array(values: np.ndarray) -> np.ndarray:
    """Copy an array into immutable bytes and rebuild its exact dtype and shape."""

    if values.dtype.hasobject:
        raise ValueError("frozen arrays must not contain Python objects")
    original_shape = values.shape
    contiguous = np.ascontiguousarray(values)
    result = np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
    ).reshape(original_shape)
    if result.flags.writeable:
        raise RuntimeError("immutable array construction produced writable storage")
    return result


def readonly_array(
    values: Any,
    *,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    """Return a defensive array whose write flag cannot be re-enabled.

    A normal owned NumPy array can be made writable again with
    ``array.setflags(write=True)`` after a caller merely clears its write flag.
    Frozen Causal4D artifacts instead retain arrays backed by immutable ``bytes``
    storage, so both direct writes and attempts to re-enable writes fail.
    """

    array = np.asarray(values, dtype=dtype)
    return _immutable_buffer_array(array)


def readonly_integer_array(values: Any, *, name: str) -> np.ndarray:
    """Return immutable ``int64`` data without coercing non-integer inputs.

    NumPy's normal ``dtype=np.int64`` conversion silently truncates floats and
    converts booleans or numeric strings. Indices are part of Causal4D's evidence
    and content-identity boundaries, so accepting those conversions could select
    a different frame, node, or support component than the caller supplied.
    """

    array = np.asarray(values)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integers")
    if array.dtype.kind == "u" and array.size:
        maximum = int(np.max(array))
        if maximum > np.iinfo(np.int64).max:
            raise ValueError(f"{name} contains an integer outside int64 range")
    return _immutable_buffer_array(array.astype(np.int64, copy=False))


__all__ = ["readonly_array", "readonly_integer_array"]
