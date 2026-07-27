"""Finite-support probability-weight utilities."""

from __future__ import annotations

import numpy as np


def log_weights_from_probabilities(
    values: np.ndarray,
    *,
    name: str = "weights",
) -> np.ndarray:
    """Convert nonnegative weights to log space without creating support.

    Exact zeros map to negative infinity. Subsequent likelihood, tempering, or
    semantic factors therefore cannot assign probability to components excluded
    by the input support.
    """

    weights = np.asarray(values, dtype=float)
    if weights.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    positive = weights > 0.0
    if not np.any(positive):
        raise ValueError(f"{name} must contain positive mass")
    result = np.full(weights.shape, -np.inf, dtype=float)
    result[positive] = np.log(weights[positive])
    return result
