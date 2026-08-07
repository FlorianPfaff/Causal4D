"""Source-only rollout-space certification for finite-support reductions.

Parameter-space moments are useful diagnostics but do not certify that a
nonlinear simulator preserves the query distribution.  This additive module
compares full and reduced support directly on a frozen source-action library.
It does not inspect target outcomes and it does not alter the registered
estimator or the existing latent-contact-v2 support decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.latent_contact_v2 import gaussian_mixture_quantiles


FUNCTIONAL_SUPPORT_CERTIFICATE_SCHEMA_VERSION = 1


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validated_string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _finite_positive_float(value: Any, *, name: str) -> float:
    result = _finite_nonnegative_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _normalized_weights(
    values: np.ndarray,
    *,
    count: int,
    name: str,
) -> np.ndarray:
    weights = readonly_array(values, dtype=float)
    if weights.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return readonly_array(weights / total, dtype=float)


@dataclass(frozen=True)
class FunctionalSupportActionV1:
    """Full and reduced predictive components for one frozen source action."""

    action_id: str
    full_trajectories_m: np.ndarray
    full_weights: np.ndarray
    reduced_trajectories_m: np.ndarray
    reduced_weights: np.ndarray
    full_component_variance_m2: np.ndarray | None = None
    reduced_component_variance_m2: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action_id = _require_nonempty_string(self.action_id, name="action_id")
        full = readonly_array(self.full_trajectories_m, dtype=float)
        reduced = readonly_array(self.reduced_trajectories_m, dtype=float)
        if full.ndim != 4 or reduced.ndim != 4:
            raise ValueError(
                "support trajectories must have shape "
                "(component, frame, node, coordinate)"
            )
        if full.shape[1:] != reduced.shape[1:]:
            raise ValueError("full and reduced trajectories must share query shape")
        if full.shape[0] < 1 or reduced.shape[0] < 1:
            raise ValueError("support trajectories must contain components")
        if not np.all(np.isfinite(full)) or not np.all(np.isfinite(reduced)):
            raise ValueError("support trajectories must be finite")
        full_weights = _normalized_weights(
            self.full_weights,
            count=full.shape[0],
            name="full_weights",
        )
        reduced_weights = _normalized_weights(
            self.reduced_weights,
            count=reduced.shape[0],
            name="reduced_weights",
        )
        full_variance = self._validated_optional_variance(
            self.full_component_variance_m2,
            full.shape,
            name="full_component_variance_m2",
        )
        reduced_variance = self._validated_optional_variance(
            self.reduced_component_variance_m2,
            reduced.shape,
            name="reduced_component_variance_m2",
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="support-action metadata must contain finite JSON data",
        )
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "full_trajectories_m", full)
        object.__setattr__(self, "full_weights", full_weights)
        object.__setattr__(self, "reduced_trajectories_m", reduced)
        object.__setattr__(self, "reduced_weights", reduced_weights)
        object.__setattr__(self, "full_component_variance_m2", full_variance)
        object.__setattr__(
            self,
            "reduced_component_variance_m2",
            reduced_variance,
        )
        object.__setattr__(self, "metadata", metadata)

    @staticmethod
    def _validated_optional_variance(
        values: np.ndarray | None,
        trajectory_shape: tuple[int, ...],
        *,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        variance = readonly_array(values, dtype=float)
        try:
            broadcast = np.broadcast_to(variance, trajectory_shape)
        except ValueError as error:
            raise ValueError(f"{name} cannot broadcast to trajectories") from error
        if not np.all(np.isfinite(broadcast)) or np.any(broadcast < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")
        return variance

    @property
    def action_artifact_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": FUNCTIONAL_SUPPORT_CERTIFICATE_SCHEMA_VERSION,
                "artifact_kind": "Causal4DFunctionalSupportActionV1",
                "action_id": self.action_id,
                "full_trajectories_sha256": array_sha256(
                    self.full_trajectories_m
                ),
                "full_weights_sha256": array_sha256(self.full_weights),
                "reduced_trajectories_sha256": array_sha256(
                    self.reduced_trajectories_m
                ),
                "reduced_weights_sha256": array_sha256(self.reduced_weights),
                "full_component_variance_sha256": (
                    None
                    if self.full_component_variance_m2 is None
                    else array_sha256(self.full_component_variance_m2)
                ),
                "reduced_component_variance_sha256": (
                    None
                    if self.reduced_component_variance_m2 is None
                    else array_sha256(self.reduced_component_variance_m2)
                ),
                "metadata": plain_json(self.metadata),
            }
        )


@dataclass(frozen=True)
class FunctionalSupportPolicyV1:
    """Source-frozen thresholds for rollout-space support preservation."""

    maximum_normalized_mean_error: float
    maximum_variance_trace_relative_error: float
    maximum_interval_endpoint_error_m: float
    maximum_energy_distance_m: float
    confidence_level: float = 0.90
    variance_floor_m2: float = 1e-12
    normalization_floor_m: float = 1e-6
    minimum_action_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_normalized_mean_error",
            _finite_nonnegative_float(
                self.maximum_normalized_mean_error,
                name="maximum_normalized_mean_error",
            ),
        )
        object.__setattr__(
            self,
            "maximum_variance_trace_relative_error",
            _finite_nonnegative_float(
                self.maximum_variance_trace_relative_error,
                name="maximum_variance_trace_relative_error",
            ),
        )
        object.__setattr__(
            self,
            "maximum_interval_endpoint_error_m",
            _finite_nonnegative_float(
                self.maximum_interval_endpoint_error_m,
                name="maximum_interval_endpoint_error_m",
            ),
        )
        object.__setattr__(
            self,
            "maximum_energy_distance_m",
            _finite_nonnegative_float(
                self.maximum_energy_distance_m,
                name="maximum_energy_distance_m",
            ),
        )
        confidence = _finite_positive_float(
            self.confidence_level,
            name="confidence_level",
        )
        if confidence >= 1.0:
            raise ValueError("confidence_level must be less than one")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "variance_floor_m2",
            _finite_positive_float(
                self.variance_floor_m2,
                name="variance_floor_m2",
            ),
        )
        object.__setattr__(
            self,
            "normalization_floor_m",
            _finite_positive_float(
                self.normalization_floor_m,
                name="normalization_floor_m",
            ),
        )
        if type(self.minimum_action_count) is not int or self.minimum_action_count < 1:
            raise ValueError("minimum_action_count must be a positive integer")


@dataclass(frozen=True)
class FunctionalSupportActionMetricsV1:
    action_id: str
    full_component_count: int
    reduced_component_count: int
    mean_rmse_m: float
    normalization_scale_m: float
    normalized_mean_error: float
    full_variance_trace_m2: float
    reduced_variance_trace_m2: float
    variance_trace_relative_error: float
    maximum_interval_endpoint_error_m: float
    energy_distance_m: float
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.action_id, name="action_id")
        if self.full_component_count < 1 or self.reduced_component_count < 1:
            raise ValueError("action metrics require positive component counts")
        for name in (
            "mean_rmse_m",
            "normalization_scale_m",
            "normalized_mean_error",
            "full_variance_trace_m2",
            "reduced_variance_trace_m2",
            "variance_trace_relative_error",
            "maximum_interval_endpoint_error_m",
            "energy_distance_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_float(getattr(self, name), name=name),
            )
        reasons = tuple(self.reasons)
        if self.accepted and reasons:
            raise ValueError("accepted action metrics cannot contain reasons")
        if not self.accepted and not reasons:
            raise ValueError("rejected action metrics require reasons")
        if len(set(reasons)) != len(reasons):
            raise ValueError("action rejection reasons must be unique")
        object.__setattr__(self, "reasons", reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "full_component_count": self.full_component_count,
            "reduced_component_count": self.reduced_component_count,
            "mean_rmse_m": self.mean_rmse_m,
            "normalization_scale_m": self.normalization_scale_m,
            "normalized_mean_error": self.normalized_mean_error,
            "full_variance_trace_m2": self.full_variance_trace_m2,
            "reduced_variance_trace_m2": self.reduced_variance_trace_m2,
            "variance_trace_relative_error": (
                self.variance_trace_relative_error
            ),
            "maximum_interval_endpoint_error_m": (
                self.maximum_interval_endpoint_error_m
            ),
            "energy_distance_m": self.energy_distance_m,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FunctionalSupportCertificateV1:
    """Fail-closed certificate over independent frozen source actions."""

    accepted: bool
    reasons: tuple[str, ...]
    action_metrics: tuple[FunctionalSupportActionMetricsV1, ...]
    policy: FunctionalSupportPolicyV1
    source_artifact_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metrics = tuple(self.action_metrics)
        if len(metrics) < self.policy.minimum_action_count:
            raise ValueError("certificate has fewer actions than required by policy")
        action_ids = tuple(metric.action_id for metric in metrics)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("certificate action IDs must be unique")
        reasons = tuple(self.reasons)
        expected_reasons = tuple(
            f"{metric.action_id}:{reason}"
            for metric in metrics
            for reason in metric.reasons
        )
        expected_accepted = not expected_reasons
        if self.accepted is not expected_accepted or reasons != expected_reasons:
            raise ValueError(
                "certificate decision must exactly match its action metrics"
            )
        if len(set(reasons)) != len(reasons):
            raise ValueError("certificate rejection reasons must be unique")
        source_ids = _validated_string_tuple(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="certificate metadata must contain finite JSON data",
        )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "action_metrics", metrics)
        object.__setattr__(self, "source_artifact_ids", source_ids)
        object.__setattr__(self, "metadata", metadata)

    @property
    def certificate_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": FUNCTIONAL_SUPPORT_CERTIFICATE_SCHEMA_VERSION,
                "artifact_kind": "Causal4DFunctionalSupportCertificateV1",
                "accepted": self.accepted,
                "reasons": list(self.reasons),
                "action_metrics": [metric.as_dict() for metric in self.action_metrics],
                "policy": asdict(self.policy),
                "source_artifact_ids": list(self.source_artifact_ids),
                "metadata": plain_json(self.metadata),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FUNCTIONAL_SUPPORT_CERTIFICATE_SCHEMA_VERSION,
            "artifact_kind": "Causal4DFunctionalSupportCertificateV1",
            "certificate_id": self.certificate_id,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "action_metrics": [metric.as_dict() for metric in self.action_metrics],
            "policy": asdict(self.policy),
            "source_artifact_ids": list(self.source_artifact_ids),
            "metadata": plain_json(self.metadata),
        }


def _component_variance(
    values: np.ndarray | None,
    trajectories: np.ndarray,
    *,
    floor_m2: float,
) -> np.ndarray:
    if values is None:
        return np.full_like(trajectories, floor_m2, dtype=float)
    return np.broadcast_to(values, trajectories.shape) + floor_m2


def _mixture_mean_and_variance(
    trajectories: np.ndarray,
    weights: np.ndarray,
    component_variance_m2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(weights[:, None, None, None] * trajectories, axis=0)
    variance = np.sum(
        weights[:, None, None, None]
        * (component_variance_m2 + np.square(trajectories - mean[None, ...])),
        axis=0,
    )
    return mean, variance


def _rms_pairwise_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_flat = first.reshape(first.shape[0], -1)
    second_flat = second.reshape(second.shape[0], -1)
    difference = first_flat[:, None, :] - second_flat[None, :, :]
    return np.sqrt(np.mean(np.square(difference), axis=-1))


def _weighted_energy_distance_m(
    full: np.ndarray,
    full_weights: np.ndarray,
    reduced: np.ndarray,
    reduced_weights: np.ndarray,
) -> float:
    cross = _rms_pairwise_distance(full, reduced)
    full_pair = _rms_pairwise_distance(full, full)
    reduced_pair = _rms_pairwise_distance(reduced, reduced)
    value = (
        2.0 * float(np.einsum("i,j,ij->", full_weights, reduced_weights, cross))
        - float(np.einsum("i,j,ij->", full_weights, full_weights, full_pair))
        - float(
            np.einsum(
                "i,j,ij->",
                reduced_weights,
                reduced_weights,
                reduced_pair,
            )
        )
    )
    return max(value, 0.0)


def _evaluate_action(
    action: FunctionalSupportActionV1,
    policy: FunctionalSupportPolicyV1,
) -> FunctionalSupportActionMetricsV1:
    full_variance = _component_variance(
        action.full_component_variance_m2,
        action.full_trajectories_m,
        floor_m2=policy.variance_floor_m2,
    )
    reduced_variance = _component_variance(
        action.reduced_component_variance_m2,
        action.reduced_trajectories_m,
        floor_m2=policy.variance_floor_m2,
    )
    full_mean, full_predictive_variance = _mixture_mean_and_variance(
        action.full_trajectories_m,
        action.full_weights,
        full_variance,
    )
    reduced_mean, reduced_predictive_variance = _mixture_mean_and_variance(
        action.reduced_trajectories_m,
        action.reduced_weights,
        reduced_variance,
    )
    mean_rmse = float(np.sqrt(np.mean(np.square(reduced_mean - full_mean))))
    normalization_scale = max(
        float(np.sqrt(np.mean(full_predictive_variance))),
        policy.normalization_floor_m,
    )
    normalized_mean_error = mean_rmse / normalization_scale
    full_trace = float(np.sum(full_predictive_variance))
    reduced_trace = float(np.sum(reduced_predictive_variance))
    trace_relative_error = abs(reduced_trace - full_trace) / max(
        full_trace,
        np.finfo(float).tiny,
    )
    tail = 0.5 * (1.0 - policy.confidence_level)
    full_interval = gaussian_mixture_quantiles(
        action.full_trajectories_m,
        full_variance,
        action.full_weights,
        (tail, 1.0 - tail),
    )
    reduced_interval = gaussian_mixture_quantiles(
        action.reduced_trajectories_m,
        reduced_variance,
        action.reduced_weights,
        (tail, 1.0 - tail),
    )
    interval_error = float(np.max(np.abs(reduced_interval - full_interval)))
    energy_distance = _weighted_energy_distance_m(
        action.full_trajectories_m,
        action.full_weights,
        action.reduced_trajectories_m,
        action.reduced_weights,
    )

    reasons: list[str] = []
    if normalized_mean_error > policy.maximum_normalized_mean_error:
        reasons.append("normalized_mean_error_exceeds_limit")
    if trace_relative_error > policy.maximum_variance_trace_relative_error:
        reasons.append("variance_trace_relative_error_exceeds_limit")
    if interval_error > policy.maximum_interval_endpoint_error_m:
        reasons.append("interval_endpoint_error_exceeds_limit")
    if energy_distance > policy.maximum_energy_distance_m:
        reasons.append("energy_distance_exceeds_limit")
    return FunctionalSupportActionMetricsV1(
        action_id=action.action_id,
        full_component_count=action.full_trajectories_m.shape[0],
        reduced_component_count=action.reduced_trajectories_m.shape[0],
        mean_rmse_m=mean_rmse,
        normalization_scale_m=normalization_scale,
        normalized_mean_error=normalized_mean_error,
        full_variance_trace_m2=full_trace,
        reduced_variance_trace_m2=reduced_trace,
        variance_trace_relative_error=trace_relative_error,
        maximum_interval_endpoint_error_m=interval_error,
        energy_distance_m=energy_distance,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def certify_functional_support_v1(
    actions: Sequence[FunctionalSupportActionV1],
    *,
    policy: FunctionalSupportPolicyV1,
    source_artifact_ids: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> FunctionalSupportCertificateV1:
    """Certify a reduction on a frozen source-action library, failing closed."""

    action_tuple = tuple(actions)
    if len(action_tuple) < policy.minimum_action_count:
        raise ValueError("insufficient source actions for functional certification")
    if any(type(action) is not FunctionalSupportActionV1 for action in action_tuple):
        raise ValueError("actions must contain FunctionalSupportActionV1 values")
    if len({action.action_id for action in action_tuple}) != len(action_tuple):
        raise ValueError("source action IDs must be unique")
    metrics = tuple(_evaluate_action(action, policy) for action in action_tuple)
    reasons = tuple(
        f"{metric.action_id}:{reason}"
        for metric in metrics
        for reason in metric.reasons
    )
    source_ids = _validated_string_tuple(
        source_artifact_ids,
        name="source_artifact_ids",
    )
    provenance = source_ids + tuple(
        action.action_artifact_id for action in action_tuple
    )
    return FunctionalSupportCertificateV1(
        accepted=not reasons,
        reasons=reasons,
        action_metrics=metrics,
        policy=policy,
        source_artifact_ids=provenance,
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "FUNCTIONAL_SUPPORT_CERTIFICATE_SCHEMA_VERSION",
    "FunctionalSupportActionMetricsV1",
    "FunctionalSupportActionV1",
    "FunctionalSupportCertificateV1",
    "FunctionalSupportPolicyV1",
    "certify_functional_support_v1",
]
