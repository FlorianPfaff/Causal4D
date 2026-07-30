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


__all__ = ["readonly_array"]
