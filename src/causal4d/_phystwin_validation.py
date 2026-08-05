"""Fail-closed validators for PhysTwin hypothesis boundaries."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from causal4d.immutable_array import readonly_array


def require_nonempty_string(value: Any, *, name: str) -> str:
    """Return one exact nonempty string or reject it."""

    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def require_exact_bool(value: Any, *, name: str) -> bool:
    """Return one exact Python boolean without truth-value coercion."""

    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def require_integer(
    value: Any,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    """Return one exact Python integer, optionally bounded below."""

    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def require_finite_real(value: Any, *, name: str) -> float:
    """Return one finite real scalar while rejecting booleans and strings."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def require_nonempty_tuple(value: Any, *, name: str) -> tuple[Any, ...]:
    """Return one exact nonempty tuple without sequence coercion."""

    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a nonempty tuple")
    return value


def require_controller_points(
    value: Any,
    *,
    name: str = "controller_points_m",
) -> np.ndarray:
    """Validate and freeze a nonempty finite ``(T, C, 3)`` control array."""

    controls = readonly_array(value, dtype=float)
    if controls.ndim != 3 or controls.shape[2] != 3:
        raise ValueError(f"{name} must have shape (T, C, 3)")
    if controls.shape[0] < 1 or controls.shape[1] < 1:
        raise ValueError(f"{name} must contain frames and controller points")
    if not np.all(np.isfinite(controls)):
        raise ValueError(f"{name} must be finite")
    return controls


def require_group_labels(
    value: Any,
    *,
    name: str,
    expected_count: int | None = None,
) -> np.ndarray:
    """Validate exact nonnegative contiguous integer controller labels."""

    raw = np.asarray(value)
    if raw.ndim != 1 or not len(raw):
        raise ValueError(f"{name} must be a nonempty vector")
    if expected_count is not None and raw.shape != (expected_count,):
        raise ValueError(f"{name} must label every controller point")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain exact integer labels")
    labels = readonly_array(raw, dtype=int)
    if np.any(labels < 0):
        raise ValueError(f"{name} must be nonnegative")
    unique = np.unique(labels)
    expected = np.arange(int(unique[-1]) + 1, dtype=int)
    if not np.array_equal(unique, expected):
        raise ValueError(f"{name} must be contiguous starting at zero")
    return labels


__all__ = [
    "require_controller_points",
    "require_exact_bool",
    "require_finite_real",
    "require_group_labels",
    "require_integer",
    "require_nonempty_string",
    "require_nonempty_tuple",
]
