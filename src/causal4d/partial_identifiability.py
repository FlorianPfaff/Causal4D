"""Prior-preserving updates for partially identified intervention supports."""

from __future__ import annotations

import numpy as np

from causal4d.identifiability import InterventionIdentifiabilityResult


def preserve_prior_within_unidentified_subspace(
    prior_weights: np.ndarray,
    updated_weights: np.ndarray,
    parameter_values: np.ndarray,
    identifiability: InterventionIdentifiabilityResult,
    *,
    grouping_tolerance: float = 1e-9,
) -> np.ndarray:
    """Remove posterior distinctions unsupported by identified directions.

    Components with the same projection onto the identified standardized
    parameter subspace retain their prior-relative probabilities. Only mass
    between distinguishable projection groups is taken from ``updated_weights``.
    A rank-zero result returns the normalized prior exactly; a full-rank result
    returns the normalized update.
    """

    prior = np.asarray(prior_weights, dtype=float).reshape(-1)
    updated = np.asarray(updated_weights, dtype=float).reshape(-1)
    values = np.asarray(parameter_values, dtype=float)
    if prior.shape != updated.shape or values.shape != (
        len(prior),
        identifiability.parameter_count,
    ):
        raise ValueError("weights and parameter_values must describe the same support")
    if not np.all(np.isfinite(prior)) or not np.all(np.isfinite(updated)):
        raise ValueError("weights must be finite")
    if np.any(prior < 0.0) or np.any(updated < 0.0):
        raise ValueError("weights must be nonnegative")
    if float(np.sum(prior)) <= 0.0 or float(np.sum(updated)) <= 0.0:
        raise ValueError("weights must contain positive mass")
    if not np.all(np.isfinite(values)):
        raise ValueError("parameter_values must be finite")
    if not np.isfinite(grouping_tolerance) or grouping_tolerance <= 0.0:
        raise ValueError("grouping_tolerance must be positive")

    prior = prior / np.sum(prior)
    updated = updated / np.sum(updated)
    if identifiability.effective_rank == 0:
        return prior.copy()
    if identifiability.effective_rank == identifiability.parameter_count:
        return updated.copy()

    standardized = values / identifiability.parameter_scales[None]
    projected = standardized @ identifiability.identified_basis
    column_scale = np.maximum(np.max(np.abs(projected), axis=0), 1.0)
    quantized = np.rint(
        projected / (grouping_tolerance * column_scale[None])
    ).astype(np.int64)

    groups: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(quantized):
        groups.setdefault(tuple(map(int, row)), []).append(index)

    result = np.zeros_like(prior)
    for indices in groups.values():
        selected = np.asarray(indices, dtype=np.int64)
        group_prior = float(np.sum(prior[selected]))
        group_updated = float(np.sum(updated[selected]))
        if group_prior <= 0.0:
            if group_updated > 0.0:
                raise ValueError("updated mass lies outside prior support")
            continue
        result[selected] = group_updated * prior[selected] / group_prior

    total = float(np.sum(result))
    if total <= 0.0:
        raise ValueError("prior-preserving update produced no mass")
    return result / total
