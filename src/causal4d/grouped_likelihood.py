"""Correlation-aware robust likelihoods over grouped observation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np

from causal4d.observation_evidence import GroupedObservationEvidence, ObservationGroup


@dataclass(frozen=True)
class GroupLikelihoodDiagnostics:
    """Nominal responsibilities and effective powers for one grouped update."""

    group_ids: tuple[str, ...]
    effective_group_weights: tuple[float, ...]
    nominal_responsibilities: np.ndarray


def _multivariate_student_t_log_density(
    residual: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    degrees_of_freedom: float,
    covariance_multiplier: float = 1.0,
) -> np.ndarray:
    """Evaluate a conventional multivariate Student-t with declared covariance."""

    values = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance_m2, dtype=float) * covariance_multiplier
    dimension = values.shape[-1]
    if covariance.shape[-2:] != (dimension, dimension):
        raise ValueError("covariance_m2 must end in (coordinate, coordinate)")
    scale = ((degrees_of_freedom - 2.0) / degrees_of_freedom) * covariance
    sign, log_determinant = np.linalg.slogdet(scale)
    if np.any(sign <= 0.0):
        raise ValueError("Student-t scale matrix must be positive definite")
    solved = np.linalg.solve(scale, values[..., None])[..., 0]
    mahalanobis = np.einsum("...i,...i->...", values, solved)
    normalization = (
        lgamma(0.5 * (degrees_of_freedom + dimension))
        - lgamma(0.5 * degrees_of_freedom)
        - 0.5
        * (
            dimension * np.log(degrees_of_freedom * np.pi)
            + log_determinant
        )
    )
    return normalization - 0.5 * (degrees_of_freedom + dimension) * np.log1p(
        mahalanobis / degrees_of_freedom
    )


def group_log_likelihood(
    predicted_values_m: np.ndarray,
    group: ObservationGroup,
    *,
    additive_variance_m2: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust mixture log likelihood and posterior nominal responsibility."""

    predictions = np.asarray(predicted_values_m, dtype=float)
    if predictions.shape[-1] != group.coordinate_count:
        raise ValueError("predicted group coordinates do not match the observation group")
    residual = predictions - group.values_m
    covariance = group.covariance_m2
    if additive_variance_m2 is not None:
        additive = np.asarray(additive_variance_m2, dtype=float)
        if additive.shape != predictions.shape:
            raise ValueError("additive_variance_m2 must match predicted group values")
        if np.any(~np.isfinite(additive)) or np.any(additive < 0.0):
            raise ValueError("additive variances must be finite and nonnegative")
        covariance = covariance + additive[..., :, None] * np.eye(group.coordinate_count)
    nominal = _multivariate_student_t_log_density(
        residual,
        covariance,
        degrees_of_freedom=group.degrees_of_freedom,
    )
    outlier = _multivariate_student_t_log_density(
        residual,
        covariance,
        degrees_of_freedom=group.degrees_of_freedom,
        covariance_multiplier=group.outlier_scale_multiplier,
    )
    log_nominal_component = np.log(group.prior_nominal_probability) + nominal
    log_outlier_component = np.log1p(-group.prior_nominal_probability) + outlier
    log_mixture = np.logaddexp(log_nominal_component, log_outlier_component)
    responsibility = np.exp(log_nominal_component - log_mixture)
    return log_mixture, responsibility


def grouped_component_log_likelihoods(
    predicted_components_m: np.ndarray,
    evidence: GroupedObservationEvidence,
    *,
    prefix_frame_count: int,
    component_variance_m2: np.ndarray | None = None,
) -> tuple[np.ndarray, GroupLikelihoodDiagnostics]:
    """Score arbitrary leading component dimensions against grouped O-plus evidence."""

    components = np.asarray(predicted_components_m, dtype=float)
    if components.ndim < 4:
        raise ValueError("predicted_components_m must end in (frame, node, coordinate)")
    if not np.all(np.isfinite(components)):
        raise ValueError("predicted components must be finite")
    evidence.validate_prefix(
        prefix_frame_count=prefix_frame_count,
        rollout_shape=components.shape[-3:],
    )
    leading_shape = components.shape[:-3]
    variance = None
    if component_variance_m2 is not None:
        variance = np.broadcast_to(
            np.asarray(component_variance_m2, dtype=float), components.shape
        )
        if np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("component variances must be finite and nonnegative")
    total = np.zeros(leading_shape, dtype=float)
    responsibilities = []
    effective_weights = evidence.effective_group_weights
    for group, weight in zip(evidence.groups, effective_weights, strict=True):
        selected = group.selected_predictions(components)
        selected_variance = (
            None if variance is None else group.selected_predictions(variance)
        )
        log_likelihood, responsibility = group_log_likelihood(
            selected, group, additive_variance_m2=selected_variance
        )
        total += weight * log_likelihood
        responsibilities.append(responsibility)
    diagnostics = GroupLikelihoodDiagnostics(
        group_ids=tuple(group.group_id for group in evidence.groups),
        effective_group_weights=effective_weights,
        nominal_responsibilities=np.stack(responsibilities, axis=-1),
    )
    return total, diagnostics


def posterior_weights_from_grouped_evidence(
    prior_weights: np.ndarray,
    predicted_components_m: np.ndarray,
    evidence: GroupedObservationEvidence,
    *,
    prefix_frame_count: int,
    component_variance_m2: np.ndarray | None = None,
) -> tuple[np.ndarray, GroupLikelihoodDiagnostics]:
    """Apply grouped evidence to finite component support in log space."""

    prior = np.asarray(prior_weights, dtype=float)
    if prior.shape != np.asarray(predicted_components_m).shape[:-3]:
        raise ValueError("prior_weights must match the component leading dimensions")
    if np.any(prior < 0.0) or not np.isclose(np.sum(prior), 1.0):
        raise ValueError("prior_weights must be nonnegative and sum to one")
    score, diagnostics = grouped_component_log_likelihoods(
        predicted_components_m,
        evidence,
        prefix_frame_count=prefix_frame_count,
        component_variance_m2=component_variance_m2,
    )
    log_posterior = np.log(np.maximum(prior, 1e-300)) + score
    maximum = float(np.max(log_posterior))
    posterior = np.exp(log_posterior - maximum)
    posterior /= np.sum(posterior)
    return posterior, diagnostics
