"""Shared normalized Student-t scores for heteroscedastic rollout likelihoods."""

from __future__ import annotations

import numpy as np


def student_t_mean_log_score(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    scale: np.ndarray | float,
    degrees_of_freedom: float,
    reduction_axes: tuple[int, ...],
    empty_error: str,
) -> np.ndarray:
    """Return a normalized mean Student-t log score.

    Constants independent of the compared component are omitted. The
    ``-log(scale)`` term is retained because conditional uncertainty may differ
    between rollout components; omitting it would reward variance inflation.
    """

    values = np.asarray(residual, dtype=float)
    scales = np.asarray(scale, dtype=float)
    if (
        not np.isfinite(degrees_of_freedom)
        or degrees_of_freedom <= 0.0
        or np.any(~np.isfinite(scales))
        or np.any(scales <= 0.0)
    ):
        raise ValueError("Student-t scales and degrees of freedom must be positive")
    try:
        standardized = values / scales
        log_scale = np.broadcast_to(np.log(scales), values.shape)
    except ValueError as error:
        raise ValueError("Student-t scale is not broadcastable to residuals") from error

    terms = -log_scale - 0.5 * (degrees_of_freedom + 1.0) * np.log1p(
        np.square(standardized) / degrees_of_freedom
    )
    valid_float = np.asarray(valid, dtype=float)
    while valid_float.ndim < terms.ndim:
        valid_float = valid_float[None]
    count = np.sum(valid_float, axis=reduction_axes)
    if np.any(count <= 0.0):
        raise ValueError(empty_error)
    return (
        np.sum(
            np.where(valid_float > 0.0, terms, 0.0),
            axis=reduction_axes,
        )
        / count
    )


__all__ = ["student_t_mean_log_score"]
