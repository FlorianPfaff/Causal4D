"""Numerical identifiability gates for intervention and nuisance responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IdentifiabilityConfig:
    """Thresholds for separating intervention from nuisance response subspaces."""

    minimum_principal_angle_degrees: float = 10.0
    maximum_projection_fraction: float = 0.95
    minimum_conditional_singular_value_ratio: float = 1e-3
    relative_rank_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_principal_angle_degrees <= 90.0:
            raise ValueError("minimum principal angle must lie in [0, 90]")
        if not 0.0 <= self.maximum_projection_fraction <= 1.0:
            raise ValueError("maximum projection fraction must lie in [0, 1]")
        if self.minimum_conditional_singular_value_ratio < 0.0:
            raise ValueError("minimum singular-value ratio must be nonnegative")
        if self.relative_rank_tolerance <= 0.0:
            raise ValueError("relative_rank_tolerance must be positive")


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Diagnostics for local linear separation of intervention and nuisance."""

    passed: bool
    intervention_rank: int
    nuisance_rank: int
    minimum_principal_angle_degrees: float
    maximum_canonical_correlation: float
    projection_fraction: float
    minimum_conditional_singular_value: float
    conditional_singular_value_ratio: float
    failed_checks: tuple[str, ...]
    config: IdentifiabilityConfig

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_checks"] = list(self.failed_checks)
        return payload


def _weighted_matrix(
    matrix: np.ndarray,
    whitening: np.ndarray | None,
    *,
    name: str,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite matrix")
    if whitening is None:
        return values
    weights = np.asarray(whitening, dtype=float)
    if weights.shape != (values.shape[0],):
        raise ValueError("whitening must contain one nonnegative row weight")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("whitening weights must be finite and nonnegative")
    if not np.any(weights > 0.0):
        raise ValueError("whitening must retain at least one response row")
    return np.sqrt(weights)[:, None] * values


def _orthonormal_basis(
    matrix: np.ndarray,
    *,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float), np.empty(0)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or singular[0] <= 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=float), singular
    rank = int(np.sum(singular > relative_tolerance * singular[0]))
    return left[:, :rank], singular[:rank]


def evaluate_response_identifiability(
    intervention_response: np.ndarray,
    nuisance_response: np.ndarray,
    *,
    whitening: np.ndarray | None = None,
    config: IdentifiabilityConfig | None = None,
) -> IdentifiabilityResult:
    """Gate a local intervention update against confounded nuisance responses."""

    settings = config or IdentifiabilityConfig()
    intervention = _weighted_matrix(
        intervention_response,
        whitening,
        name="intervention_response",
    )
    nuisance = _weighted_matrix(
        nuisance_response,
        whitening,
        name="nuisance_response",
    )
    if intervention.shape[0] != nuisance.shape[0]:
        raise ValueError("intervention and nuisance responses must share rows")
    intervention_basis, intervention_singular = _orthonormal_basis(
        intervention,
        relative_tolerance=settings.relative_rank_tolerance,
    )
    nuisance_basis, _ = _orthonormal_basis(
        nuisance,
        relative_tolerance=settings.relative_rank_tolerance,
    )
    intervention_rank = intervention_basis.shape[1]
    nuisance_rank = nuisance_basis.shape[1]
    if intervention_rank == 0:
        raise ValueError("intervention response has zero numerical rank")

    if nuisance_rank == 0:
        maximum_correlation = 0.0
        minimum_angle = 90.0
        projected = np.zeros_like(intervention)
    else:
        correlations = np.linalg.svd(
            intervention_basis.T @ nuisance_basis,
            compute_uv=False,
        )
        maximum_correlation = float(np.clip(correlations[0], 0.0, 1.0))
        minimum_angle = float(np.degrees(np.arccos(maximum_correlation)))
        projected = nuisance_basis @ (nuisance_basis.T @ intervention)

    intervention_norm = float(np.linalg.norm(intervention, ord="fro"))
    projection_fraction = float(
        np.linalg.norm(projected, ord="fro") / intervention_norm
    )
    conditional = intervention - projected
    conditional_singular = np.linalg.svd(conditional, compute_uv=False)
    minimum_conditional = float(
        conditional_singular[min(intervention_rank, len(conditional_singular)) - 1]
    )
    reference_singular = float(intervention_singular[0])
    conditional_ratio = minimum_conditional / reference_singular

    failed = []
    if minimum_angle < settings.minimum_principal_angle_degrees:
        failed.append("principal_angle")
    if projection_fraction > settings.maximum_projection_fraction:
        failed.append("projection_fraction")
    if conditional_ratio < settings.minimum_conditional_singular_value_ratio:
        failed.append("conditional_singular_value")
    return IdentifiabilityResult(
        passed=not failed,
        intervention_rank=intervention_rank,
        nuisance_rank=nuisance_rank,
        minimum_principal_angle_degrees=minimum_angle,
        maximum_canonical_correlation=maximum_correlation,
        projection_fraction=projection_fraction,
        minimum_conditional_singular_value=minimum_conditional,
        conditional_singular_value_ratio=conditional_ratio,
        failed_checks=tuple(failed),
        config=settings,
    )
