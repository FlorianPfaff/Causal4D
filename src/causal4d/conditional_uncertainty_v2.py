"""Prospective structured conditional uncertainty for latent-contact v2.

The registered estimator and the existing :mod:`causal4d.latent_contact_v2`
rollout-bank contract remain unchanged. This additive module supplies a
provenance-bearing uncertainty layer that can be projected through the existing
linear observation groups and can expose the full law-of-total-covariance result
for joint calibration diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.baselines import PredictiveDistribution
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.latent_contact_v2 import (
    ContactLikelihoodV2Diagnostics,
    ContactObservationEvidenceV2,
    ContactPatchRolloutBankV2,
    gaussian_mixture_quantiles,
    posterior_weights_from_contact_evidence_v2,
)


CONDITIONAL_UNCERTAINTY_V2_SCHEMA_VERSION = 1
DEFAULT_MAXIMUM_JOINT_DIMENSION = 2048


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


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _normalized_weights(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.shape != expected_shape:
        raise ValueError(f"joint_weights must have shape {expected_shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("joint_weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("joint_weights must contain positive mass")
    return weights / total


@dataclass(frozen=True)
class ConditionalPredictiveUncertaintyV2:
    """Diagonal residual variance plus Gaussian low-rank trajectory modes.

    ``independent_variance_m2`` must broadcast to the rollout-bank trajectory
    shape ``(state, particle, frame, node, coordinate)``. It is added to the
    bank's existing scalar variance floor.

    ``low_rank_factors_m`` may be shared or component-specific. Its final four
    axes are ``(rank, frame, node, coordinate)``; any leading axes must broadcast
    to ``(state, particle)``. For one component, factors ``F`` define the
    correlated conditional covariance ``F.T @ F`` after flattening the queried
    trajectory coordinates.
    """

    source_artifact_ids: tuple[str, ...]
    independent_variance_m2: np.ndarray = field(
        default_factory=lambda: np.asarray(0.0, dtype=float)
    )
    low_rank_factors_m: np.ndarray | None = None
    uncertainty_id: str = "conditional_predictive_uncertainty_v2"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_ids = _validated_string_tuple(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )
        independent = readonly_array(self.independent_variance_m2, dtype=float)
        if not np.all(np.isfinite(independent)) or np.any(independent < 0.0):
            raise ValueError("independent_variance_m2 must be finite and nonnegative")
        factors = None
        if self.low_rank_factors_m is not None:
            factors = readonly_array(self.low_rank_factors_m, dtype=float)
            if factors.ndim < 4 or factors.shape[-4] < 1:
                raise ValueError(
                    "low_rank_factors_m must end in "
                    "(positive rank, frame, node, coordinate)"
                )
            if not np.all(np.isfinite(factors)):
                raise ValueError("low_rank_factors_m must be finite")
        uncertainty_id = _require_nonempty_string(
            self.uncertainty_id,
            name="uncertainty_id",
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="uncertainty metadata must contain finite JSON data",
        )
        object.__setattr__(self, "source_artifact_ids", source_ids)
        object.__setattr__(self, "independent_variance_m2", independent)
        object.__setattr__(self, "low_rank_factors_m", factors)
        object.__setattr__(self, "uncertainty_id", uncertainty_id)
        object.__setattr__(self, "metadata", metadata)

    @property
    def artifact_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": CONDITIONAL_UNCERTAINTY_V2_SCHEMA_VERSION,
                "artifact_kind": "Causal4DConditionalPredictiveUncertaintyV2",
                "uncertainty_id": self.uncertainty_id,
                "source_artifact_ids": list(self.source_artifact_ids),
                "independent_variance_sha256": array_sha256(
                    self.independent_variance_m2
                ),
                "low_rank_factors_sha256": (
                    None
                    if self.low_rank_factors_m is None
                    else array_sha256(self.low_rank_factors_m)
                ),
                "metadata": plain_json(self.metadata),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONDITIONAL_UNCERTAINTY_V2_SCHEMA_VERSION,
            "artifact_kind": "Causal4DConditionalPredictiveUncertaintyV2",
            "artifact_id": self.artifact_id,
            "uncertainty_id": self.uncertainty_id,
            "source_artifact_ids": list(self.source_artifact_ids),
            "independent_variance_sha256": array_sha256(self.independent_variance_m2),
            "low_rank_factors_sha256": (
                None
                if self.low_rank_factors_m is None
                else array_sha256(self.low_rank_factors_m)
            ),
            "metadata": plain_json(self.metadata),
        }

    def independent_component_variance_m2(
        self,
        bank: ContactPatchRolloutBankV2,
    ) -> np.ndarray:
        """Return broadcast residual variance including the bank noise floor."""

        target_shape = bank.trajectories_m.shape
        try:
            variance = np.broadcast_to(
                np.asarray(self.independent_variance_m2, dtype=float),
                target_shape,
            )
        except ValueError as error:
            raise ValueError(
                "independent_variance_m2 cannot broadcast to rollout components"
            ) from error
        result = variance + bank.variance_floor_m2
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
            raise ValueError("combined independent component variance must be positive")
        return result

    def component_low_rank_factors_m(
        self,
        bank: ContactPatchRolloutBankV2,
    ) -> np.ndarray | None:
        """Broadcast low-rank factors to every state/parameter component."""

        if self.low_rank_factors_m is None:
            return None
        rank = self.low_rank_factors_m.shape[-4]
        target_shape = (
            *bank.trajectories_m.shape[:-3],
            rank,
            *bank.trajectories_m.shape[-3:],
        )
        try:
            return np.broadcast_to(self.low_rank_factors_m, target_shape)
        except ValueError as error:
            raise ValueError(
                "low_rank_factors_m cannot broadcast to rollout components"
            ) from error

    def marginal_component_variance_m2(
        self,
        bank: ContactPatchRolloutBankV2,
    ) -> np.ndarray:
        """Return each Gaussian component's coordinatewise marginal variance."""

        variance = self.independent_component_variance_m2(bank).copy()
        factors = self.component_low_rank_factors_m(bank)
        if factors is not None:
            variance += np.sum(np.square(factors), axis=-4)
        return variance

    def component_group_covariance_m2(
        self,
        bank: ContactPatchRolloutBankV2,
        evidence: ContactObservationEvidenceV2,
    ) -> dict[str, np.ndarray]:
        """Project correlated modes through every existing linear group."""

        factors = self.component_low_rank_factors_m(bank)
        if factors is None:
            return {}
        result: dict[str, np.ndarray] = {}
        for group in evidence.groups:
            projected = group.apply(factors)
            result[group.group_id] = np.einsum(
                "...ri,...rj->...ij",
                projected,
                projected,
            )
        return result


@dataclass(frozen=True)
class JointPredictiveMomentsV2:
    """Full joint trajectory covariance for calibration and NEES diagnostics."""

    method: str
    mean: np.ndarray
    covariance_m2: np.ndarray
    source_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        method = _require_nonempty_string(self.method, name="method")
        mean = readonly_array(self.mean, dtype=float)
        covariance = readonly_array(self.covariance_m2, dtype=float)
        source_ids = _validated_string_tuple(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )
        if mean.ndim != 3:
            raise ValueError(
                "joint predictive mean must have shape (frame, node, coordinate)"
            )
        dimension = int(np.prod(mean.shape))
        if covariance.shape != (dimension, dimension):
            raise ValueError(
                "joint predictive covariance must match the flattened mean"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("joint predictive moments must be finite")
        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("joint predictive covariance must be symmetric")
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if minimum_eigenvalue < -1e-10:
            raise ValueError(
                "joint predictive covariance must be positive semidefinite"
            )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance_m2", covariance)
        object.__setattr__(self, "source_artifact_ids", source_ids)

    @property
    def variance_m2(self) -> np.ndarray:
        diagonal = np.diag(self.covariance_m2).reshape(self.mean.shape)
        return readonly_array(diagonal, dtype=float)

    @property
    def artifact_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": CONDITIONAL_UNCERTAINTY_V2_SCHEMA_VERSION,
                "artifact_kind": "Causal4DJointPredictiveMomentsV2",
                "method": self.method,
                "mean_sha256": array_sha256(self.mean),
                "covariance_sha256": array_sha256(self.covariance_m2),
                "source_artifact_ids": list(self.source_artifact_ids),
            }
        )


def _combined_group_covariances(
    structured: Mapping[str, np.ndarray],
    additional: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value, dtype=float) for key, value in structured.items()}
    for key, value in dict(additional or {}).items():
        additive = np.asarray(value, dtype=float)
        if key in result:
            try:
                result[key] = result[key] + additive
            except ValueError as error:
                raise ValueError(
                    f"additional covariance for group {key!r} cannot broadcast"
                ) from error
        else:
            result[key] = additive
    return result


def posterior_weights_with_conditional_uncertainty_v2(
    bank: ContactPatchRolloutBankV2,
    evidence: ContactObservationEvidenceV2,
    uncertainty: ConditionalPredictiveUncertaintyV2,
    *,
    prefix_frame_count: int,
    additional_group_covariance_m2: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, ContactLikelihoodV2Diagnostics]:
    """Update joint component weights with diagonal and correlated uncertainty."""

    group_covariance = _combined_group_covariances(
        uncertainty.component_group_covariance_m2(bank, evidence),
        additional_group_covariance_m2,
    )
    return posterior_weights_from_contact_evidence_v2(
        bank.prior_joint_weights,
        bank.trajectories_m,
        evidence,
        prefix_frame_count=prefix_frame_count,
        component_variance_m2=(uncertainty.independent_component_variance_m2(bank)),
        component_group_covariance_m2=group_covariance,
    )


def predictive_distribution_with_conditional_uncertainty_v2(
    bank: ContactPatchRolloutBankV2,
    uncertainty: ConditionalPredictiveUncertaintyV2,
    joint_weights: np.ndarray | None = None,
    *,
    method: str = "latent_contact_patch_v2_structured_uncertainty",
    conditional_variance_multiplier: float = 1.0,
    include_intervals: bool = True,
) -> PredictiveDistribution:
    """Mix component-specific marginal Gaussian uncertainty exactly."""

    weights = _normalized_weights(
        bank.prior_joint_weights if joint_weights is None else joint_weights,
        expected_shape=bank.prior_joint_weights.shape,
    )
    multiplier = _finite_positive_float(
        conditional_variance_multiplier,
        name="conditional_variance_multiplier",
    )
    flat_weights = weights.reshape(-1)
    components = bank.trajectories_m.reshape(
        -1,
        *bank.trajectories_m.shape[-3:],
    )
    conditional_variance = multiplier * uncertainty.marginal_component_variance_m2(
        bank
    ).reshape(components.shape)
    mean = np.sum(flat_weights[:, None, None, None] * components, axis=0)
    variance = np.sum(
        flat_weights[:, None, None, None]
        * (conditional_variance + np.square(components - mean[None, ...])),
        axis=0,
    )
    if not include_intervals:
        return PredictiveDistribution(method=method, mean=mean, variance=variance)
    tail = 0.5 * (1.0 - bank.confidence_level)
    interval = gaussian_mixture_quantiles(
        components,
        conditional_variance,
        flat_weights,
        (tail, 1.0 - tail),
    )
    return PredictiveDistribution(
        method=method,
        mean=mean,
        variance=variance,
        interval_lower=interval[0],
        interval_upper=interval[1],
    )


def joint_predictive_moments_with_conditional_uncertainty_v2(
    bank: ContactPatchRolloutBankV2,
    uncertainty: ConditionalPredictiveUncertaintyV2,
    joint_weights: np.ndarray | None = None,
    *,
    method: str = "latent_contact_patch_v2_structured_uncertainty",
    conditional_variance_multiplier: float = 1.0,
    maximum_joint_dimension: int = DEFAULT_MAXIMUM_JOINT_DIMENSION,
) -> JointPredictiveMomentsV2:
    """Apply the full law of total covariance in flattened trajectory space.

    This diagnostic intentionally materializes a dense ``D x D`` covariance.
    ``maximum_joint_dimension`` is checked before any quadratic allocation and
    must be raised explicitly for larger bounded diagnostics.
    """

    dimension = int(np.prod(bank.trajectories_m.shape[-3:]))
    maximum_dimension = _positive_integer(
        maximum_joint_dimension,
        name="maximum_joint_dimension",
    )
    if dimension > maximum_dimension:
        dense_bytes = dimension * dimension * np.dtype(float).itemsize
        raise ValueError(
            "joint predictive dimension exceeds maximum_joint_dimension: "
            f"dimension={dimension}, maximum={maximum_dimension}, "
            f"dense_covariance_bytes={dense_bytes}"
        )
    weights = _normalized_weights(
        bank.prior_joint_weights if joint_weights is None else joint_weights,
        expected_shape=bank.prior_joint_weights.shape,
    ).reshape(-1)
    multiplier = _finite_positive_float(
        conditional_variance_multiplier,
        name="conditional_variance_multiplier",
    )
    components = bank.trajectories_m.reshape(-1, dimension)
    mean_flat = np.sum(weights[:, None] * components, axis=0)
    centered = components - mean_flat[None, :]
    covariance = np.einsum("k,ki,kj->ij", weights, centered, centered)

    independent = uncertainty.independent_component_variance_m2(bank).reshape(
        components.shape
    )
    covariance += np.diag(multiplier * np.sum(weights[:, None] * independent, axis=0))
    factors = uncertainty.component_low_rank_factors_m(bank)
    if factors is not None:
        flat_factors = factors.reshape(
            len(weights),
            factors.shape[-4],
            components.shape[1],
        )
        covariance += multiplier * np.einsum(
            "k,kri,krj->ij",
            weights,
            flat_factors,
            flat_factors,
        )
    covariance = 0.5 * (covariance + covariance.T)
    return JointPredictiveMomentsV2(
        method=method,
        mean=mean_flat.reshape(bank.trajectories_m.shape[-3:]),
        covariance_m2=covariance,
        source_artifact_ids=(bank.bank_id, uncertainty.artifact_id),
    )


__all__ = [
    "CONDITIONAL_UNCERTAINTY_V2_SCHEMA_VERSION",
    "DEFAULT_MAXIMUM_JOINT_DIMENSION",
    "ConditionalPredictiveUncertaintyV2",
    "JointPredictiveMomentsV2",
    "joint_predictive_moments_with_conditional_uncertainty_v2",
    "posterior_weights_with_conditional_uncertainty_v2",
    "predictive_distribution_with_conditional_uncertainty_v2",
]
