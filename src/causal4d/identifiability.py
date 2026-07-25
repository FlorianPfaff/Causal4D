"""Intervention-versus-nuisance identifiability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class IdentifiabilityConfig:
    """Frozen thresholds for conditional intervention information."""

    relative_rank_tolerance: float = 1e-6
    minimum_information_eigenvalue: float = 1e-6
    maximum_condition_number: float = 1e8
    minimum_residualized_response_fraction: float = 0.10
    maximum_subspace_cosine: float = 0.995

    def __post_init__(self) -> None:
        if not 0.0 < self.relative_rank_tolerance < 1.0:
            raise ValueError("relative_rank_tolerance must lie in (0, 1)")
        if self.minimum_information_eigenvalue <= 0.0:
            raise ValueError("minimum_information_eigenvalue must be positive")
        if self.maximum_condition_number <= 1.0:
            raise ValueError("maximum_condition_number must exceed one")
        if not 0.0 <= self.minimum_residualized_response_fraction <= 1.0:
            raise ValueError("minimum_residualized_response_fraction must lie in [0, 1]")
        if not 0.0 <= self.maximum_subspace_cosine <= 1.0:
            raise ValueError("maximum_subspace_cosine must lie in [0, 1]")


@dataclass(frozen=True)
class InterventionIdentifiabilityResult:
    """Conditional information remaining after projecting out nuisance response."""

    conditional_information: np.ndarray
    eigenvalues: np.ndarray
    effective_rank: int
    parameter_count: int
    minimum_eigenvalue: float
    condition_number: float
    residualized_response_fraction: float
    maximum_subspace_cosine: float
    identifiable: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        information = np.asarray(self.conditional_information, dtype=float).copy()
        eigenvalues = np.asarray(self.eigenvalues, dtype=float).copy()
        if information.shape != (self.parameter_count, self.parameter_count):
            raise ValueError("conditional_information must match parameter_count")
        if eigenvalues.shape != (self.parameter_count,):
            raise ValueError("eigenvalues must match parameter_count")
        if not np.all(np.isfinite(information)) or not np.all(np.isfinite(eigenvalues)):
            raise ValueError("identifiability arrays must be finite")
        information.setflags(write=False)
        eigenvalues.setflags(write=False)
        object.__setattr__(self, "conditional_information", information)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))

    def as_dict(self) -> dict[str, object]:
        return {
            "effective_rank": self.effective_rank,
            "parameter_count": self.parameter_count,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "condition_number": (
                self.condition_number if np.isfinite(self.condition_number) else None
            ),
            "residualized_response_fraction": self.residualized_response_fraction,
            "maximum_subspace_cosine": self.maximum_subspace_cosine,
            "identifiable": self.identifiable,
            "failure_reasons": list(self.failure_reasons),
            "eigenvalues": self.eigenvalues.tolist(),
        }


def finite_response_sensitivity(
    reference_response: np.ndarray,
    perturbed_responses: np.ndarray,
    perturbation_steps: Sequence[float],
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Build a flattened secant-sensitivity matrix from finite perturbations.

    ``perturbed_responses`` has shape ``(parameter, ...)`` and each remaining
    dimension must match ``reference_response``. A boolean ``valid`` mask may
    select response coordinates before flattening.
    """

    reference = np.asarray(reference_response, dtype=float)
    perturbed = np.asarray(perturbed_responses, dtype=float)
    steps = np.asarray(tuple(perturbation_steps), dtype=float)
    if perturbed.ndim != reference.ndim + 1 or perturbed.shape[1:] != reference.shape:
        raise ValueError("perturbed_responses must have shape (parameter, *reference.shape)")
    if steps.shape != (len(perturbed),) or np.any(~np.isfinite(steps)) or np.any(steps == 0.0):
        raise ValueError("perturbation_steps must be finite, nonzero, and match parameters")
    responses = (perturbed - reference[None]) / steps.reshape((-1,) + (1,) * reference.ndim)
    if valid is None:
        selected = np.ones(reference.shape, dtype=bool)
    else:
        selected = np.asarray(valid, dtype=bool)
        if selected.shape != reference.shape:
            raise ValueError("valid must match reference_response")
    if not np.any(selected):
        raise ValueError("finite-response sensitivity has no valid coordinates")
    matrix = responses[:, selected].T
    if not np.all(np.isfinite(matrix)):
        raise ValueError("finite-response sensitivities must be finite")
    return matrix


def _whiten(matrix: np.ndarray, covariance: np.ndarray | None) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("sensitivity matrices must be nonempty two-dimensional arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("sensitivity matrices must be finite")
    if covariance is None:
        return values
    noise = np.asarray(covariance, dtype=float)
    if noise.shape != (values.shape[0], values.shape[0]):
        raise ValueError("covariance must match the response dimension")
    if not np.all(np.isfinite(noise)) or not np.allclose(noise, noise.T, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    try:
        factor = np.linalg.cholesky(noise)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error
    return np.linalg.solve(factor, values)


def _orthonormal_basis(matrix: np.ndarray, tolerance: float) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] == 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    rank = int(np.sum(singular_values > tolerance * singular_values[0]))
    return left[:, :rank]


def assess_intervention_identifiability(
    intervention_sensitivity: np.ndarray,
    nuisance_sensitivity: np.ndarray | None = None,
    *,
    covariance: np.ndarray | None = None,
    config: IdentifiabilityConfig | None = None,
) -> InterventionIdentifiabilityResult:
    """Assess intervention information conditional on nuisance response.

    The supplied sensitivities are whitened by ``covariance``. Intervention
    columns are then projected onto the orthogonal complement of the nuisance
    response. The resulting Gram matrix is the local conditional information.
    """

    settings = config or IdentifiabilityConfig()
    intervention = _whiten(intervention_sensitivity, covariance)
    response_count, parameter_count = intervention.shape
    if nuisance_sensitivity is None:
        nuisance = np.zeros((response_count, 0), dtype=float)
    else:
        nuisance_raw = np.asarray(nuisance_sensitivity, dtype=float)
        if nuisance_raw.ndim != 2 or nuisance_raw.shape[0] != response_count:
            raise ValueError("nuisance_sensitivity must share the response dimension")
        nuisance = _whiten(nuisance_raw, covariance)

    nuisance_basis = _orthonormal_basis(nuisance, settings.relative_rank_tolerance)
    intervention_basis = _orthonormal_basis(
        intervention, settings.relative_rank_tolerance
    )
    residualized = intervention - nuisance_basis @ (nuisance_basis.T @ intervention)
    information = residualized.T @ residualized
    information = 0.5 * (information + information.T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(information), 0.0)
    largest = float(eigenvalues[-1])
    tolerance = settings.relative_rank_tolerance * max(largest, 1.0)
    effective_rank = int(np.sum(eigenvalues > tolerance))
    minimum = float(eigenvalues[0])
    positive = eigenvalues[eigenvalues > tolerance]
    condition_number = (
        float(np.max(positive) / np.min(positive)) if len(positive) else float("inf")
    )
    original_energy = float(np.sum(np.square(intervention)))
    residual_energy = float(np.sum(np.square(residualized)))
    residual_fraction = residual_energy / original_energy if original_energy > 0.0 else 0.0
    if nuisance_basis.shape[1] and intervention_basis.shape[1]:
        maximum_cosine = float(
            np.max(np.linalg.svd(nuisance_basis.T @ intervention_basis, compute_uv=False))
        )
    else:
        maximum_cosine = 0.0

    reasons = []
    if effective_rank < parameter_count:
        reasons.append("rank_deficient_after_nuisance_projection")
    if minimum < settings.minimum_information_eigenvalue:
        reasons.append("conditional_information_below_threshold")
    if condition_number > settings.maximum_condition_number:
        reasons.append("conditional_information_ill_conditioned")
    if residual_fraction < settings.minimum_residualized_response_fraction:
        reasons.append("intervention_response_absorbed_by_nuisance")
    if maximum_cosine > settings.maximum_subspace_cosine:
        reasons.append("intervention_and_nuisance_subspaces_nearly_collinear")
    return InterventionIdentifiabilityResult(
        conditional_information=information,
        eigenvalues=eigenvalues,
        effective_rank=effective_rank,
        parameter_count=parameter_count,
        minimum_eigenvalue=minimum,
        condition_number=condition_number,
        residualized_response_fraction=float(residual_fraction),
        maximum_subspace_cosine=maximum_cosine,
        identifiable=not reasons,
        failure_reasons=tuple(reasons),
    )
