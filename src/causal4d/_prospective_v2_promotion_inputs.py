"""Target-free inputs for the prospective V2 promotion experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from causal4d._prospective_v2_promotion_common import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    _finite_float,
    _finite_nonnegative_float,
    _rate,
    _require_no_target_access,
    _require_nonempty_string,
    _require_sha256,
    _validated_source_metadata,
    _validated_string_tuple,
)
from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS
from causal4d.immutable_json import plain_json


@dataclass(frozen=True)
class ProspectiveV2CandidateV1:
    """One predeclared member of the fixed V2 candidate ladder."""

    candidate_id: str
    candidate_kind: str
    artifact_id: str
    source_configuration_id: str
    components: tuple[str, ...]
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = _require_nonempty_string(
            self.candidate_id,
            name="candidate_id",
        )
        candidate_kind = _require_nonempty_string(
            self.candidate_kind,
            name="candidate_kind",
        )
        if candidate_kind not in PROSPECTIVE_V2_CANDIDATE_KINDS:
            raise ValueError(
                "candidate_kind must belong to the prospective V2 ladder"
            )
        artifact_id = _require_sha256(self.artifact_id, name="artifact_id")
        source_configuration_id = _require_sha256(
            self.source_configuration_id,
            name="source_configuration_id",
        )
        components = _validated_string_tuple(self.components, name="components")
        target_outcomes_used = _require_no_target_access(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        metadata = _validated_source_metadata(
            self.metadata,
            name="candidate metadata",
        )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "candidate_kind", candidate_kind)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(
            self,
            "source_configuration_id",
            source_configuration_id,
        )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(self, "metadata", metadata)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "artifact_id": self.artifact_id,
            "source_configuration_id": self.source_configuration_id,
            "components": list(self.components),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True)
class ProspectiveV2EvaluationUnitV1:
    """One registered independent session/seed unit for one endpoint."""

    unit_id: str
    endpoint: str
    independent_group_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_id = _require_nonempty_string(self.unit_id, name="unit_id")
        endpoint = _require_nonempty_string(self.endpoint, name="endpoint")
        if endpoint not in DECISION_TRACE_ENDPOINTS:
            raise ValueError(
                f"endpoint must be one of {list(DECISION_TRACE_ENDPOINTS)}"
            )
        independent_group_id = _require_nonempty_string(
            self.independent_group_id,
            name="independent_group_id",
        )
        metadata = _validated_source_metadata(
            self.metadata,
            name="evaluation-unit metadata",
        )
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "independent_group_id", independent_group_id)
        object.__setattr__(self, "metadata", metadata)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "endpoint": self.endpoint,
            "independent_group_id": self.independent_group_id,
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True)
class ProspectiveV2PromotionPolicyV1:
    """Predeclared endpoint-wise promotion thresholds."""

    minimum_units_per_endpoint: int
    minimum_mean_log_score_gain: float
    maximum_mean_brier_change: float
    maximum_mean_trajectory_regret_m: float
    maximum_mean_coverage_error: float
    maximum_mean_interval_width_ratio: float
    minimum_accepted_update_rate: float
    maximum_harmful_accepted_update_rate: float
    maximum_fallback_rate: float
    interval_width_floor_m: float = 1e-9

    def __post_init__(self) -> None:
        if (
            type(self.minimum_units_per_endpoint) is not int
            or self.minimum_units_per_endpoint < 1
        ):
            raise ValueError("minimum_units_per_endpoint must be a positive integer")
        for name in (
            "minimum_mean_log_score_gain",
            "maximum_mean_brier_change",
            "maximum_mean_trajectory_regret_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "maximum_mean_coverage_error",
            _rate(
                self.maximum_mean_coverage_error,
                name="maximum_mean_coverage_error",
            ),
        )
        object.__setattr__(
            self,
            "maximum_mean_interval_width_ratio",
            _finite_nonnegative_float(
                self.maximum_mean_interval_width_ratio,
                name="maximum_mean_interval_width_ratio",
            ),
        )
        minimum_accepted_update_rate = _rate(
            self.minimum_accepted_update_rate,
            name="minimum_accepted_update_rate",
        )
        if minimum_accepted_update_rate <= 0.0:
            raise ValueError("minimum_accepted_update_rate must be positive")
        object.__setattr__(
            self,
            "minimum_accepted_update_rate",
            minimum_accepted_update_rate,
        )
        object.__setattr__(
            self,
            "maximum_harmful_accepted_update_rate",
            _rate(
                self.maximum_harmful_accepted_update_rate,
                name="maximum_harmful_accepted_update_rate",
            ),
        )
        maximum_fallback_rate = _rate(
            self.maximum_fallback_rate,
            name="maximum_fallback_rate",
        )
        if maximum_fallback_rate >= 1.0:
            raise ValueError("maximum_fallback_rate must be less than one")
        object.__setattr__(
            self,
            "maximum_fallback_rate",
            maximum_fallback_rate,
        )
        floor = _finite_nonnegative_float(
            self.interval_width_floor_m,
            name="interval_width_floor_m",
        )
        if floor <= 0.0:
            raise ValueError("interval_width_floor_m must be positive")
        object.__setattr__(self, "interval_width_floor_m", floor)
