"""Candidate-level aggregate for prospective V2 promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causal4d._prospective_v2_promotion_common import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    _require_bool,
    _require_nonempty_string,
    _require_sha256,
)
from causal4d._prospective_v2_promotion_metrics import (
    ProspectiveV2EndpointMetricsV1,
)
from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS


@dataclass(frozen=True)
class ProspectiveV2CandidateResultV1:
    candidate_id: str
    candidate_kind: str
    candidate_artifact_id: str
    endpoint_metrics: tuple[ProspectiveV2EndpointMetricsV1, ...]
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        candidate_id = _require_nonempty_string(
            self.candidate_id,
            name="candidate_id",
        )
        candidate_kind = _require_nonempty_string(
            self.candidate_kind,
            name="candidate_kind",
        )
        if candidate_kind not in PROSPECTIVE_V2_CANDIDATE_KINDS[1:]:
            raise ValueError("candidate result must describe a non-baseline candidate")
        artifact_id = _require_sha256(
            self.candidate_artifact_id,
            name="candidate_artifact_id",
        )
        endpoint_metrics = tuple(self.endpoint_metrics)
        if any(
            type(metric) is not ProspectiveV2EndpointMetricsV1
            for metric in endpoint_metrics
        ):
            raise ValueError(
                "endpoint_metrics must contain ProspectiveV2EndpointMetricsV1 values"
            )
        if tuple(metric.endpoint for metric in endpoint_metrics) != (
            DECISION_TRACE_ENDPOINTS
        ):
            raise ValueError("candidate result must cover every endpoint in order")
        if any(metric.candidate_id != candidate_id for metric in endpoint_metrics):
            raise ValueError("endpoint metrics must identify their candidate")
        expected_reasons = tuple(
            f"{metric.endpoint}:{reason}"
            for metric in endpoint_metrics
            for reason in metric.reasons
        )
        accepted = _require_bool(self.accepted, name="accepted")
        reasons = tuple(self.reasons)
        expected_accepted = not expected_reasons
        if reasons != expected_reasons or accepted is not expected_accepted:
            raise ValueError(
                "candidate result decision must exactly match endpoint metrics"
            )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "candidate_kind", candidate_kind)
        object.__setattr__(self, "candidate_artifact_id", artifact_id)
        object.__setattr__(self, "endpoint_metrics", endpoint_metrics)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "reasons", reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "candidate_artifact_id": self.candidate_artifact_id,
            "endpoint_metrics": [
                metric.as_dict() for metric in self.endpoint_metrics
            ],
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }
