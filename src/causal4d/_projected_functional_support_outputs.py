"""Output contracts for task-projected functional-support certification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from causal4d._projected_functional_support_common import (
    PROJECTED_FUNCTIONAL_SUPPORT_SCHEMA_VERSION,
    canonical_sha256,
    finite_nonnegative_float,
    require_no_target_access,
    require_nonempty_string,
    require_sha256,
    validated_source_metadata,
    validated_string_tuple,
)
from causal4d._projected_functional_support_inputs import (
    ProjectedFunctionalSupportPolicyV1,
)
from causal4d.immutable_json import plain_json


@dataclass(frozen=True)
class ProjectedFunctionalSupportMetricV1:
    action_id: str
    projection_id: str
    full_mean_m: float
    reduced_mean_m: float
    full_variance_m2: float
    reduced_variance_m2: float
    projected_variance_relative_error: float
    maximum_projected_interval_endpoint_error_m: float
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_id",
            require_nonempty_string(self.action_id, name="action_id"),
        )
        object.__setattr__(
            self,
            "projection_id",
            require_nonempty_string(self.projection_id, name="projection_id"),
        )
        for name in ("full_mean_m", "reduced_mean_m"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                raise ValueError(f"{name} must be finite")
            result = float(value)
            if not np.isfinite(result):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, result)
        for name in (
            "full_variance_m2",
            "reduced_variance_m2",
            "projected_variance_relative_error",
            "maximum_projected_interval_endpoint_error_m",
        ):
            object.__setattr__(
                self,
                name,
                finite_nonnegative_float(getattr(self, name), name=name),
            )
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        reasons = tuple(self.reasons)
        if self.accepted and reasons:
            raise ValueError("accepted projected metrics cannot contain reasons")
        if not self.accepted and not reasons:
            raise ValueError("rejected projected metrics require reasons")
        if len(set(reasons)) != len(reasons):
            raise ValueError("projected metric reasons must be unique")
        object.__setattr__(self, "reasons", reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "projection_id": self.projection_id,
            "full_mean_m": self.full_mean_m,
            "reduced_mean_m": self.reduced_mean_m,
            "full_variance_m2": self.full_variance_m2,
            "reduced_variance_m2": self.reduced_variance_m2,
            "projected_variance_relative_error": (
                self.projected_variance_relative_error
            ),
            "maximum_projected_interval_endpoint_error_m": (
                self.maximum_projected_interval_endpoint_error_m
            ),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ProjectedFunctionalSupportCertificateV1:
    """Fail-closed projection certificate layered on the base certificate."""

    accepted: bool
    reasons: tuple[str, ...]
    metrics: tuple[ProjectedFunctionalSupportMetricV1, ...]
    policy: ProjectedFunctionalSupportPolicyV1
    base_certificate_id: str
    base_certificate_accepted: bool
    base_certificate_reasons: tuple[str, ...]
    base_action_artifact_ids: tuple[str, ...]
    action_artifact_ids: tuple[str, ...]
    projection_artifact_ids: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        if not metrics or any(
            type(metric) is not ProjectedFunctionalSupportMetricV1 for metric in metrics
        ):
            raise ValueError(
                "metrics must contain ProjectedFunctionalSupportMetricV1 values"
            )
        metric_keys = tuple(
            (metric.action_id, metric.projection_id) for metric in metrics
        )
        if len(set(metric_keys)) != len(metric_keys):
            raise ValueError("projected metric pairs must be unique")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        reasons = tuple(self.reasons)
        base_id = require_sha256(
            self.base_certificate_id,
            name="base_certificate_id",
        )
        if type(self.base_certificate_accepted) is not bool:
            raise ValueError("base_certificate_accepted must be a boolean")
        base_reasons = tuple(self.base_certificate_reasons)
        if self.base_certificate_accepted and base_reasons:
            raise ValueError("accepted base certificate cannot contain reasons")
        if not self.base_certificate_accepted and not base_reasons:
            raise ValueError("rejected base certificate requires reasons")
        base_action_ids = validated_string_tuple(
            self.base_action_artifact_ids,
            name="base_action_artifact_ids",
            require_digest=True,
        )
        action_ids = validated_string_tuple(
            self.action_artifact_ids,
            name="action_artifact_ids",
            require_digest=True,
        )
        projection_ids = validated_string_tuple(
            self.projection_artifact_ids,
            name="projection_artifact_ids",
            require_digest=True,
        )
        action_names = tuple(dict.fromkeys(metric.action_id for metric in metrics))
        projection_names = tuple(
            dict.fromkeys(metric.projection_id for metric in metrics)
        )
        if len(base_action_ids) != len(action_ids):
            raise ValueError("base and projected action counts must match")
        if len(action_names) != len(action_ids):
            raise ValueError("metrics must cover every action exactly")
        if len(projection_names) != len(projection_ids):
            raise ValueError("metrics must cover every projection exactly")
        expected_keys = tuple(
            (action_name, projection_name)
            for action_name in action_names
            for projection_name in projection_names
        )
        if metric_keys != expected_keys:
            raise ValueError("metrics must follow the full action/projection product")
        expected_reasons = tuple(f"base:{reason}" for reason in base_reasons) + tuple(
            f"{metric.action_id}:{metric.projection_id}:{reason}"
            for metric in metrics
            for reason in metric.reasons
        )
        expected_accepted = not expected_reasons
        if reasons != expected_reasons or self.accepted is not expected_accepted:
            raise ValueError("certificate decision must exactly match its evidence")
        if len(set(reasons)) != len(reasons):
            raise ValueError("certificate reasons must be unique")
        source_ids = validated_string_tuple(
            self.source_artifact_ids,
            name="source_artifact_ids",
            require_digest=True,
        )
        target_used = require_no_target_access(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        metadata = validated_source_metadata(
            self.metadata,
            name="projected certificate metadata",
        )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "base_certificate_id", base_id)
        object.__setattr__(self, "base_certificate_reasons", base_reasons)
        object.__setattr__(self, "base_action_artifact_ids", base_action_ids)
        object.__setattr__(self, "action_artifact_ids", action_ids)
        object.__setattr__(self, "projection_artifact_ids", projection_ids)
        object.__setattr__(self, "source_artifact_ids", source_ids)
        object.__setattr__(self, "target_outcomes_used", target_used)
        object.__setattr__(self, "metadata", metadata)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROJECTED_FUNCTIONAL_SUPPORT_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProjectedFunctionalSupportCertificateV1",
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "metrics": [metric.as_dict() for metric in self.metrics],
            "policy": asdict(self.policy),
            "base_certificate_id": self.base_certificate_id,
            "base_certificate_accepted": self.base_certificate_accepted,
            "base_certificate_reasons": list(self.base_certificate_reasons),
            "base_action_artifact_ids": list(self.base_action_artifact_ids),
            "action_artifact_ids": list(self.action_artifact_ids),
            "projection_artifact_ids": list(self.projection_artifact_ids),
            "source_artifact_ids": list(self.source_artifact_ids),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def certificate_id(self) -> str:
        return canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "certificate_id": self.certificate_id}
