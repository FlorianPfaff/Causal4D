"""Input contracts for task-projected functional-support certification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from causal4d._projected_functional_support_common import (
    PROJECTED_FUNCTIONAL_SUPPORT_SCHEMA_VERSION,
    canonical_sha256,
    finite_nonnegative_float,
    finite_positive_float,
    require_nonempty_string,
    validated_source_metadata,
)
from causal4d.contracts import array_sha256
from causal4d.functional_support_v1 import FunctionalSupportActionV1
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json


@dataclass(frozen=True)
class FunctionalSupportProjectionV1:
    """One immutable linear task readout over a rollout trajectory."""

    projection_id: str
    coefficients: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        projection_id = require_nonempty_string(
            self.projection_id,
            name="projection_id",
        )
        coefficients = readonly_array(self.coefficients, dtype=float)
        if coefficients.ndim != 3 or coefficients.size < 1:
            raise ValueError(
                "projection coefficients must have shape (frame, node, coordinate)"
            )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("projection coefficients must be finite")
        if not np.any(coefficients != 0.0):
            raise ValueError("projection coefficients must contain a nonzero value")
        metadata = validated_source_metadata(
            self.metadata,
            name="projection metadata",
        )
        object.__setattr__(self, "projection_id", projection_id)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "metadata", metadata)

    @property
    def projection_artifact_id(self) -> str:
        return canonical_sha256(
            {
                "schema_version": PROJECTED_FUNCTIONAL_SUPPORT_SCHEMA_VERSION,
                "artifact_kind": "Causal4DFunctionalSupportProjectionV1",
                "projection_id": self.projection_id,
                "coefficients_sha256": array_sha256(self.coefficients),
                "metadata": plain_json(self.metadata),
            }
        )


@dataclass(frozen=True)
class ProjectedFunctionalSupportActionV1:
    """One base action plus optional component-specific low-rank modes."""

    action: FunctionalSupportActionV1
    full_component_low_rank_factors_m: np.ndarray | None = None
    reduced_component_low_rank_factors_m: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.action) is not FunctionalSupportActionV1:
            raise ValueError("action must be a FunctionalSupportActionV1")
        full_factors = self._validated_factors(
            self.full_component_low_rank_factors_m,
            self.action.full_trajectories_m.shape,
            name="full_component_low_rank_factors_m",
        )
        reduced_factors = self._validated_factors(
            self.reduced_component_low_rank_factors_m,
            self.action.reduced_trajectories_m.shape,
            name="reduced_component_low_rank_factors_m",
        )
        metadata = validated_source_metadata(
            self.metadata,
            name="projected action metadata",
        )
        object.__setattr__(
            self,
            "full_component_low_rank_factors_m",
            full_factors,
        )
        object.__setattr__(
            self,
            "reduced_component_low_rank_factors_m",
            reduced_factors,
        )
        object.__setattr__(self, "metadata", metadata)

    @staticmethod
    def _validated_factors(
        values: np.ndarray | None,
        trajectory_shape: tuple[int, ...],
        *,
        name: str,
    ) -> np.ndarray | None:
        if values is None:
            return None
        factors = readonly_array(values, dtype=float)
        if factors.ndim not in (4, 5) or factors.shape[-4] < 1:
            raise ValueError(
                f"{name} must end in "
                "(positive rank, frame, node, coordinate) with an optional "
                "leading component axis"
            )
        rank = factors.shape[-4]
        target_shape = (trajectory_shape[0], rank, *trajectory_shape[1:])
        try:
            broadcast = np.broadcast_to(factors, target_shape)
        except ValueError as error:
            raise ValueError(
                f"{name} cannot broadcast to component trajectories"
            ) from error
        if not np.all(np.isfinite(broadcast)):
            raise ValueError(f"{name} must be finite")
        return readonly_array(broadcast, dtype=float)

    @property
    def action_id(self) -> str:
        return self.action.action_id

    @property
    def projected_action_artifact_id(self) -> str:
        return canonical_sha256(
            {
                "schema_version": PROJECTED_FUNCTIONAL_SUPPORT_SCHEMA_VERSION,
                "artifact_kind": "Causal4DProjectedFunctionalSupportActionV1",
                "base_action_artifact_id": self.action.action_artifact_id,
                "full_low_rank_factors_sha256": (
                    None
                    if self.full_component_low_rank_factors_m is None
                    else array_sha256(self.full_component_low_rank_factors_m)
                ),
                "reduced_low_rank_factors_sha256": (
                    None
                    if self.reduced_component_low_rank_factors_m is None
                    else array_sha256(self.reduced_component_low_rank_factors_m)
                ),
                "metadata": plain_json(self.metadata),
            }
        )


@dataclass(frozen=True)
class ProjectedFunctionalSupportPolicyV1:
    """Source-frozen gates for task-projected predictive dependence."""

    maximum_projected_variance_relative_error: float
    maximum_projected_interval_endpoint_error_m: float
    confidence_level: float = 0.90
    variance_floor_m2: float = 1e-12
    minimum_projection_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "maximum_projected_variance_relative_error",
            finite_nonnegative_float(
                self.maximum_projected_variance_relative_error,
                name="maximum_projected_variance_relative_error",
            ),
        )
        object.__setattr__(
            self,
            "maximum_projected_interval_endpoint_error_m",
            finite_nonnegative_float(
                self.maximum_projected_interval_endpoint_error_m,
                name="maximum_projected_interval_endpoint_error_m",
            ),
        )
        confidence = finite_positive_float(
            self.confidence_level,
            name="confidence_level",
        )
        if confidence >= 1.0:
            raise ValueError("confidence_level must be less than one")
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self,
            "variance_floor_m2",
            finite_positive_float(
                self.variance_floor_m2,
                name="variance_floor_m2",
            ),
        )
        if (
            type(self.minimum_projection_count) is not int
            or self.minimum_projection_count < 1
        ):
            raise ValueError("minimum_projection_count must be a positive integer")
