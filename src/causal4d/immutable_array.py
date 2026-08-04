"""Owned read-only NumPy array helpers for frozen artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np


def readonly_array(
    values: Any,
    *,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    """Return a defensive NumPy copy that rejects writes."""

    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def readonly_integer_array(values: Any, *, name: str) -> np.ndarray:
    """Return owned ``int64`` data without coercing non-integer inputs.

    NumPy's normal ``dtype=np.int64`` conversion silently truncates floats and
    converts booleans or numeric strings.  Indices are part of Causal4D's
    evidence and content-identity boundaries, so accepting those conversions
    could select a different frame, node, or support component than the caller
    supplied.
    """

    array = np.asarray(values)
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integers")
    if array.dtype.kind == "u" and array.size:
        maximum = int(np.max(array))
        if maximum > np.iinfo(np.int64).max:
            raise ValueError(f"{name} contains an integer outside int64 range")
    result = array.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result


__all__ = ["readonly_array", "readonly_integer_array"]
