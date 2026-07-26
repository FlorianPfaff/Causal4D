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
            raise ValueError(
                "minimum_residualized_response_fraction must lie in [0, 1]"
            )
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
    parameter_scale: np.ndarray | None = None
    identifiable_basis: np.ndarray | None = None
    nullspace_basis: np.ndarray | None = None

    def __post_init__(self) -> None:
        information = np.asarray(self.conditional_information, dtype=float).copy()
        eigenvalues = np.asarray(self.eigenvalues, dtype=float).copy()
        if information.shape != (self.parameter_count, self.parameter_count):
            raise ValueError("conditional_information must match parameter_count")
        if eigenvalues.shape != (self.parameter_count,):
            raise ValueError("eigenvalues must match parameter_count")
        if not np.all(np.isfinite(information)) or not np.all(np.isfinite(eigenvalues)):
            raise ValueError("identifiability arrays must be finite")

        if self.parameter_scale is None:
            scale = np.ones(self.parameter_count, dtype=float)
        else:
            scale = np.asarray(self.parameter_scale, dtype=float).copy()
        if scale.shape != (self.parameter_count,) or not np.all(np.isfinite(scale)):
            raise ValueError("parameter_scale must match parameter_count")
        if np.any(scale <= 0.0):
            raise ValueError("parameter_scale must be strictly positive")

        if self.identifiable_basis is None:
            identifiable_basis = np.zeros((self.parameter_count, 0), dtype=float)
        else:
            identifiable_basis = np.asarray(self.identifiable_basis, dtype=float).copy()
        if identifiable_basis.shape != (self.parameter_count, self.effective_rank):
            raise ValueError("identifiable_basis must match effective_rank")

        nullity = self.parameter_count - self.effective_rank
        if self.nullspace_basis is None:
            nullspace_basis = np.zeros((self.parameter_count, nullity), dtype=float)
        else:
            nullspace_basis = np.asarray(self.nullspace_basis, dtype=float).copy()
        if nullspace_basis.shape != (self.parameter_count, nullity):
            raise ValueError("nullspace_basis must match the information nullity")
        if not np.all(np.isfinite(identifiable_basis)) or not np.all(
            np.isfinite(nullspace_basis)
        ):
            raise ValueError("identifiability bases must be finite")

        for array in (
            information,
            eigenvalues,
            scale,
            identifiable_basis,
            nullspace_basis,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "conditional_information", information)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "parameter_scale", scale)
        object.__setattr__(self, "identifiable_basis", identifiable_basis)
        object.__setattr__(self, "nullspace_basis", nullspace_basis)
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
            "parameter_scale": self.parameter_scale.tolist(),
            "identifiable_basis": self.identifiable_basis.tolist(),
        }

    def project_parameter_values(self, values: np.ndarray) -> np.ndarray:
        """Project finite parameter support into the identifiable standardized span."""

        supplied = np.asarray(values, dtype=float)
        if supplied.ndim != 2 or supplied.shape[1] != self.parameter_count:
            raise ValueError("values must have shape (component, parameter_count)")
        if not np.all(np.isfinite(supplied)):
            raise ValueError("parameter values must be finite")
        standardized = supplied / self.parameter_scale[None]
        return standardized @ self.identifiable_basis


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
        raise ValueError(
            "perturbed_responses must have shape (parameter, *reference.shape)"
        )
    if (
        steps.shape != (len(perturbed),)
        or np.any(~np.isfinite(steps))
        or np.any(steps == 0.0)
    ):
        raise ValueError(
            "perturbation_steps must be finite, nonzero, and match parameters"
        )
    responses = (perturbed - reference[None]) / steps.reshape(
        (-1,) + (1,) * reference.ndim
    )
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
    parameter_scale: Sequence[float] | np.ndarray | None = None,
    config: IdentifiabilityConfig | None = None,
) -> InterventionIdentifiabilityResult:
    """Assess intervention information conditional on nuisance response.

    ``parameter_scale`` defines source-frozen characteristic parameter changes.
    Sensitivity columns are multiplied by this scale before information is
    evaluated, making rank and conditioning invariant to unit changes such as
    degrees versus radians or frames versus seconds.
    """

    settings = config or IdentifiabilityConfig()
    raw_intervention = np.asarray(intervention_sensitivity, dtype=float)
    if raw_intervention.ndim != 2:
        raise ValueError("intervention_sensitivity must be two-dimensional")
    parameter_count = raw_intervention.shape[1]
    if parameter_scale is None:
        scale = np.ones(parameter_count, dtype=float)
    else:
        scale = np.asarray(tuple(parameter_scale), dtype=float)
        if scale.shape != (parameter_count,) or not np.all(np.isfinite(scale)):
            raise ValueError("parameter_scale must match intervention parameters")
        if np.any(scale <= 0.0):
            raise ValueError("parameter_scale must be strictly positive")
    intervention = _whiten(raw_intervention * scale[None], covariance)
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
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[-1])
    tolerance = settings.relative_rank_tolerance * max(largest, 1.0)
    selected = eigenvalues > tolerance
    effective_rank = int(np.sum(selected))
    identifiable_basis = eigenvectors[:, selected]
    nullspace_basis = eigenvectors[:, ~selected]
    minimum = float(eigenvalues[0])
    positive = eigenvalues[selected]
    condition_number = (
        float(np.max(positive) / np.min(positive)) if len(positive) else float("inf")
    )
    original_energy = float(np.sum(np.square(intervention)))
    residual_energy = float(np.sum(np.square(residualized)))
    residual_fraction = (
        residual_energy / original_energy if original_energy > 0.0 else 0.0
    )
    if nuisance_basis.shape[1] and intervention_basis.shape[1]:
        maximum_cosine = float(
            np.max(
                np.linalg.svd(
                    nuisance_basis.T @ intervention_basis, compute_uv=False
                )
            )
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
        parameter_scale=scale,
        identifiable_basis=identifiable_basis,
        nullspace_basis=nullspace_basis,
    )


def preserve_prior_within_unidentified_subspace(
    prior_weights: np.ndarray,
    updated_weights: np.ndarray,
    parameter_values: np.ndarray,
    identifiability: InterventionIdentifiabilityResult,
    *,
    grouping_tolerance: float = 1e-9,
) -> np.ndarray:
    """Remove unsupported posterior distinctions along unidentified directions.

    Components with the same identifiable projection retain their original
    prior-relative probabilities. Only mass between distinguishable projection
    groups is taken from ``updated_weights``. A rank-zero result returns the
    normalized prior exactly; a full-rank result returns the normalized update.
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
    if not np.isfinite(grouping_tolerance) or grouping_tolerance <= 0.0:
        raise ValueError("grouping_tolerance must be positive")
    prior = prior / np.sum(prior)
    updated = updated / np.sum(updated)
    if identifiability.effective_rank == 0:
        return prior.copy()
    if identifiability.effective_rank == identifiability.parameter_count:
        return updated.copy()

    projected = identifiability.project_parameter_values(values)
    column_scale = np.maximum(np.max(np.abs(projected), axis=0), 1.0)
    quantized = np.rint(
        projected / (grouping_tolerance * column_scale[None])
    ).astype(np.int64)
    group_lookup: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(quantized):
        group_lookup.setdefault(tuple(map(int, row)), []).append(index)

    result = np.zeros_like(prior)
    for indices in group_lookup.values():
        selected = np.asarray(indices, dtype=np.int64)
        group_prior = float(np.sum(prior[selected]))
        group_updated = float(np.sum(updated[selected]))
        if group_prior <= 0.0:
            if group_updated > 0.0:
                raise ValueError("updated mass lies outside prior support")
            continue
        result[selected] = group_updated * prior[selected] / group_prior
    result /= np.sum(result)
    return result
