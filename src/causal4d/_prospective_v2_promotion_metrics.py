"""Independent-unit and endpoint metrics for prospective V2 promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from causal4d._prospective_v2_promotion_common import (
    PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
    _canonical_sha256,
    _finite_float,
    _finite_nonnegative_float,
    _rate,
    _require_bool,
    _require_nonempty_string,
    _require_sha256,
    _require_target_access,
)
from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS
from causal4d.immutable_json import plain_json, validated_json_mapping


@dataclass(frozen=True)
class ProspectiveV2UnitMetricsV1:
    """One candidate outcome on one registered independent unit."""

    unit_id: str
    endpoint: str
    candidate_id: str
    evaluation_opening_id: str
    log_score: float
    brier_score: float
    trajectory_error_m: float
    coverage_error: float
    interval_width_m: float
    candidate_accepted: bool
    harmful_update: bool
    fallback_used: bool
    target_outcomes_used: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_id = _require_nonempty_string(self.unit_id, name="unit_id")
        endpoint = _require_nonempty_string(self.endpoint, name="endpoint")
        if endpoint not in DECISION_TRACE_ENDPOINTS:
            raise ValueError("unit metric endpoint is invalid")
        candidate_id = _require_nonempty_string(
            self.candidate_id,
            name="candidate_id",
        )
        opening_id = _require_sha256(
            self.evaluation_opening_id,
            name="evaluation_opening_id",
        )
        log_score = _finite_float(self.log_score, name="log_score")
        brier_score = _finite_nonnegative_float(
            self.brier_score,
            name="brier_score",
        )
        trajectory_error = _finite_nonnegative_float(
            self.trajectory_error_m,
            name="trajectory_error_m",
        )
        coverage_error = _rate(self.coverage_error, name="coverage_error")
        interval_width = _finite_nonnegative_float(
            self.interval_width_m,
            name="interval_width_m",
        )
        accepted = _require_bool(
            self.candidate_accepted,
            name="candidate_accepted",
        )
        harmful = _require_bool(self.harmful_update, name="harmful_update")
        fallback = _require_bool(self.fallback_used, name="fallback_used")
        if harmful and not accepted:
            raise ValueError("harmful_update requires an accepted candidate update")
        if fallback is accepted:
            raise ValueError(
                "fallback_used must be the exact complement of candidate_accepted"
            )
        target_outcomes_used = _require_target_access(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="unit-metric metadata must contain finite JSON data",
        )
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "evaluation_opening_id", opening_id)
        object.__setattr__(self, "log_score", log_score)
        object.__setattr__(self, "brier_score", brier_score)
        object.__setattr__(self, "trajectory_error_m", trajectory_error)
        object.__setattr__(self, "coverage_error", coverage_error)
        object.__setattr__(self, "interval_width_m", interval_width)
        object.__setattr__(self, "candidate_accepted", accepted)
        object.__setattr__(self, "harmful_update", harmful)
        object.__setattr__(self, "fallback_used", fallback)
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(self, "metadata", metadata)

    @property
    def metric_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
                "artifact_kind": "Causal4DProspectiveV2UnitMetricsV1",
                **self.as_dict(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "endpoint": self.endpoint,
            "candidate_id": self.candidate_id,
            "evaluation_opening_id": self.evaluation_opening_id,
            "log_score": self.log_score,
            "brier_score": self.brier_score,
            "trajectory_error_m": self.trajectory_error_m,
            "coverage_error": self.coverage_error,
            "interval_width_m": self.interval_width_m,
            "candidate_accepted": self.candidate_accepted,
            "harmful_update": self.harmful_update,
            "fallback_used": self.fallback_used,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True)
class ProspectiveV2EndpointMetricsV1:
    candidate_id: str
    endpoint: str
    unit_count: int
    mean_log_score_gain: float
    mean_brier_change: float
    mean_trajectory_regret_m: float
    mean_coverage_error: float
    mean_interval_width_m: float
    mean_interval_width_ratio: float
    accepted_update_rate: float
    harmful_accepted_update_rate: float
    fallback_rate: float
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _require_nonempty_string(self.candidate_id, name="candidate_id"),
        )
        endpoint = _require_nonempty_string(self.endpoint, name="endpoint")
        if endpoint not in DECISION_TRACE_ENDPOINTS:
            raise ValueError("endpoint aggregate has an invalid endpoint")
        object.__setattr__(self, "endpoint", endpoint)
        if type(self.unit_count) is not int or self.unit_count < 1:
            raise ValueError("unit_count must be a positive integer")
        for name in (
            "mean_log_score_gain",
            "mean_brier_change",
            "mean_trajectory_regret_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )
        for name in (
            "mean_coverage_error",
            "mean_interval_width_m",
            "mean_interval_width_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_float(getattr(self, name), name=name),
            )
        for name in (
            "accepted_update_rate",
            "harmful_accepted_update_rate",
            "fallback_rate",
        ):
            object.__setattr__(self, name, _rate(getattr(self, name), name=name))
        accepted = _require_bool(self.accepted, name="accepted")
        reasons = tuple(self.reasons)
        if accepted and reasons:
            raise ValueError("accepted endpoint metrics cannot contain reasons")
        if not accepted and not reasons:
            raise ValueError("rejected endpoint metrics require reasons")
        if len(set(reasons)) != len(reasons):
            raise ValueError("endpoint rejection reasons must be unique")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "reasons", reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "endpoint": self.endpoint,
            "unit_count": self.unit_count,
            "mean_log_score_gain": self.mean_log_score_gain,
            "mean_brier_change": self.mean_brier_change,
            "mean_trajectory_regret_m": self.mean_trajectory_regret_m,
            "mean_coverage_error": self.mean_coverage_error,
            "mean_interval_width_m": self.mean_interval_width_m,
            "mean_interval_width_ratio": self.mean_interval_width_ratio,
            "accepted_update_rate": self.accepted_update_rate,
            "harmful_accepted_update_rate": self.harmful_accepted_update_rate,
            "fallback_rate": self.fallback_rate,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }
