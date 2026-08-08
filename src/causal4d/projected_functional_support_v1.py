"""Source-only task-projection certification for support reductions.

The additive certificate checks frozen linear task readouts after the existing
rollout-space functional-support certificate. It preserves the registered and
frozen estimators and never reads target outcomes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from causal4d._projected_functional_support_common import (
    PROJECTED_FUNCTIONAL_SUPPORT_SCHEMA_VERSION,
    validated_string_tuple,
)
from causal4d._projected_functional_support_inputs import (
    FunctionalSupportProjectionV1,
    ProjectedFunctionalSupportActionV1,
    ProjectedFunctionalSupportPolicyV1,
)
from causal4d._projected_functional_support_outputs import (
    ProjectedFunctionalSupportCertificateV1,
    ProjectedFunctionalSupportMetricV1,
)
from causal4d.functional_support_v1 import (
    FunctionalSupportActionV1,
    FunctionalSupportCertificateV1,
)
from causal4d.latent_contact_v2 import gaussian_mixture_quantiles


def _component_variance(
    values: np.ndarray | None,
    trajectories: np.ndarray,
    *,
    floor_m2: float,
) -> np.ndarray:
    if values is None:
        return np.full_like(trajectories, floor_m2, dtype=float)
    return np.broadcast_to(values, trajectories.shape) + floor_m2


def _projected_component_moments(
    trajectories: np.ndarray,
    component_variance_m2: np.ndarray,
    low_rank_factors_m: np.ndarray | None,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.einsum(
        "ifnc,fnc->i",
        trajectories,
        coefficients,
        optimize=True,
    )
    variances = np.einsum(
        "ifnc,fnc->i",
        component_variance_m2,
        np.square(coefficients),
        optimize=True,
    )
    if low_rank_factors_m is not None:
        projected_factors = np.einsum(
            "irfnc,fnc->ir",
            low_rank_factors_m,
            coefficients,
            optimize=True,
        )
        variances = variances + np.sum(np.square(projected_factors), axis=1)
    if (
        not np.all(np.isfinite(means))
        or not np.all(np.isfinite(variances))
        or np.any(variances < 0.0)
    ):
        raise ValueError("projected component moments must be finite and valid")
    return means, variances


def _mixture_moments(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    mixture_mean = float(np.sum(weights * means))
    mixture_variance = float(
        np.sum(weights * (variances + np.square(means - mixture_mean)))
    )
    if not np.isfinite(mixture_mean) or not np.isfinite(mixture_variance):
        raise ValueError("projected mixture moments must be finite")
    if mixture_variance < 0.0:
        raise ValueError("projected mixture variance must be nonnegative")
    return mixture_mean, mixture_variance


def _mixture_interval(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    *,
    confidence_level: float,
) -> np.ndarray:
    tail = 0.5 * (1.0 - confidence_level)
    interval = gaussian_mixture_quantiles(
        means[:, None, None, None],
        variances[:, None, None, None],
        weights,
        (tail, 1.0 - tail),
    )
    result = np.asarray(interval[:, 0, 0, 0], dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError("projected mixture interval must be finite")
    return result


def _evaluate_projection(
    projected_action: ProjectedFunctionalSupportActionV1,
    projection: FunctionalSupportProjectionV1,
    policy: ProjectedFunctionalSupportPolicyV1,
) -> ProjectedFunctionalSupportMetricV1:
    action = projected_action.action
    query_shape = action.full_trajectories_m.shape[1:]
    if projection.coefficients.shape != query_shape:
        raise ValueError(
            f"projection {projection.projection_id!r} has shape "
            f"{projection.coefficients.shape}, expected {query_shape} for "
            f"action {action.action_id!r}"
        )
    full_diagonal = _component_variance(
        action.full_component_variance_m2,
        action.full_trajectories_m,
        floor_m2=policy.variance_floor_m2,
    )
    reduced_diagonal = _component_variance(
        action.reduced_component_variance_m2,
        action.reduced_trajectories_m,
        floor_m2=policy.variance_floor_m2,
    )
    full_means, full_variances = _projected_component_moments(
        action.full_trajectories_m,
        full_diagonal,
        projected_action.full_component_low_rank_factors_m,
        projection.coefficients,
    )
    reduced_means, reduced_variances = _projected_component_moments(
        action.reduced_trajectories_m,
        reduced_diagonal,
        projected_action.reduced_component_low_rank_factors_m,
        projection.coefficients,
    )
    full_mean, full_variance = _mixture_moments(
        full_means,
        full_variances,
        action.full_weights,
    )
    reduced_mean, reduced_variance = _mixture_moments(
        reduced_means,
        reduced_variances,
        action.reduced_weights,
    )
    variance_error = abs(reduced_variance - full_variance) / max(
        full_variance,
        np.finfo(float).tiny,
    )
    full_interval = _mixture_interval(
        full_means,
        full_variances,
        action.full_weights,
        confidence_level=policy.confidence_level,
    )
    reduced_interval = _mixture_interval(
        reduced_means,
        reduced_variances,
        action.reduced_weights,
        confidence_level=policy.confidence_level,
    )
    interval_error = float(np.max(np.abs(reduced_interval - full_interval)))
    reasons: list[str] = []
    if variance_error > policy.maximum_projected_variance_relative_error:
        reasons.append("projected_variance_relative_error_exceeds_limit")
    if interval_error > policy.maximum_projected_interval_endpoint_error_m:
        reasons.append("projected_interval_endpoint_error_exceeds_limit")
    return ProjectedFunctionalSupportMetricV1(
        action_id=action.action_id,
        projection_id=projection.projection_id,
        full_mean_m=full_mean,
        reduced_mean_m=reduced_mean,
        full_variance_m2=full_variance,
        reduced_variance_m2=reduced_variance,
        projected_variance_relative_error=variance_error,
        maximum_projected_interval_endpoint_error_m=interval_error,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def _coerce_action(
    value: FunctionalSupportActionV1 | ProjectedFunctionalSupportActionV1,
) -> ProjectedFunctionalSupportActionV1:
    if type(value) is ProjectedFunctionalSupportActionV1:
        return value
    if type(value) is FunctionalSupportActionV1:
        return ProjectedFunctionalSupportActionV1(action=value)
    raise ValueError(
        "actions must contain FunctionalSupportActionV1 or "
        "ProjectedFunctionalSupportActionV1 values"
    )


def certify_projected_functional_support_v1(
    actions: Sequence[
        FunctionalSupportActionV1 | ProjectedFunctionalSupportActionV1
    ],
    projections: Sequence[FunctionalSupportProjectionV1],
    *,
    policy: ProjectedFunctionalSupportPolicyV1,
    base_certificate: FunctionalSupportCertificateV1,
    source_artifact_ids: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> ProjectedFunctionalSupportCertificateV1:
    """Certify task projections after the base rollout-space certificate."""

    action_tuple = tuple(_coerce_action(action) for action in actions)
    projection_tuple = tuple(projections)
    if not action_tuple:
        raise ValueError("actions must be nonempty")
    if len({action.action_id for action in action_tuple}) != len(action_tuple):
        raise ValueError("source action IDs must be unique")
    if any(action.action.target_outcomes_used for action in action_tuple):
        raise ValueError("source actions must not use target outcomes")
    if len(projection_tuple) < policy.minimum_projection_count:
        raise ValueError("insufficient task projections for certification")
    if any(
        type(projection) is not FunctionalSupportProjectionV1
        for projection in projection_tuple
    ):
        raise ValueError(
            "projections must contain FunctionalSupportProjectionV1 values"
        )
    if len({projection.projection_id for projection in projection_tuple}) != len(
        projection_tuple
    ):
        raise ValueError("projection IDs must be unique")
    if type(base_certificate) is not FunctionalSupportCertificateV1:
        raise ValueError("base_certificate must be a FunctionalSupportCertificateV1")
    action_ids = tuple(action.action_id for action in action_tuple)
    base_action_ids = tuple(
        metric.action_id for metric in base_certificate.action_metrics
    )
    if base_action_ids != action_ids:
        raise ValueError(
            "base certificate action order must match projected source actions"
        )
    base_action_artifact_ids = tuple(
        action.action.action_artifact_id for action in action_tuple
    )
    missing_provenance = tuple(
        artifact_id
        for artifact_id in base_action_artifact_ids
        if artifact_id not in base_certificate.source_artifact_ids
    )
    if missing_provenance:
        raise ValueError("base certificate does not bind every projected action")
    metrics = tuple(
        _evaluate_projection(action, projection, policy)
        for action in action_tuple
        for projection in projection_tuple
    )
    reasons = tuple(
        f"base:{reason}" for reason in base_certificate.reasons
    ) + tuple(
        f"{metric.action_id}:{metric.projection_id}:{reason}"
        for metric in metrics
        for reason in metric.reasons
    )
    source_ids = validated_string_tuple(
        source_artifact_ids,
        name="source_artifact_ids",
    )
    return ProjectedFunctionalSupportCertificateV1(
        accepted=not reasons,
        reasons=reasons,
        metrics=metrics,
        policy=policy,
        base_certificate_id=base_certificate.certificate_id,
        base_certificate_accepted=base_certificate.accepted,
        base_certificate_reasons=base_certificate.reasons,
        base_action_artifact_ids=base_action_artifact_ids,
        action_artifact_ids=tuple(
            action.projected_action_artifact_id for action in action_tuple
        ),
        projection_artifact_ids=tuple(
            projection.projection_artifact_id for projection in projection_tuple
        ),
        source_artifact_ids=source_ids,
        target_outcomes_used=False,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "PROJECTED_FUNCTIONAL_SUPPORT_SCHEMA_VERSION",
    "FunctionalSupportProjectionV1",
    "ProjectedFunctionalSupportActionV1",
    "ProjectedFunctionalSupportCertificateV1",
    "ProjectedFunctionalSupportMetricV1",
    "ProjectedFunctionalSupportPolicyV1",
    "certify_projected_functional_support_v1",
]
