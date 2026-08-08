"""Content-addressed target-free freeze for prospective V2 promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from causal4d._prospective_v2_promotion_common import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
    _canonical_sha256,
    _require_no_target_access,
    _require_nonempty_string,
    _require_sha256,
    _validated_source_metadata,
    _validated_string_tuple,
)
from causal4d._prospective_v2_promotion_inputs import (
    ProspectiveV2CandidateV1,
    ProspectiveV2EvaluationUnitV1,
    ProspectiveV2PromotionPolicyV1,
)
from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS
from causal4d.immutable_json import plain_json


@dataclass(frozen=True)
class ProspectiveV2PromotionFreezeV1:
    """Target-free freeze for one untouched V2 promotion panel."""

    experiment_id: str
    stack_lock_id: str
    candidates: tuple[ProspectiveV2CandidateV1, ...]
    evaluation_units: tuple[ProspectiveV2EvaluationUnitV1, ...]
    policy: ProspectiveV2PromotionPolicyV1
    source_artifact_ids: tuple[str, ...]
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        experiment_id = _require_nonempty_string(
            self.experiment_id,
            name="experiment_id",
        )
        stack_lock_id = _require_sha256(
            self.stack_lock_id,
            name="stack_lock_id",
        )
        if type(self.policy) is not ProspectiveV2PromotionPolicyV1:
            raise ValueError("policy must be a ProspectiveV2PromotionPolicyV1")
        candidates = tuple(self.candidates)
        if any(type(value) is not ProspectiveV2CandidateV1 for value in candidates):
            raise ValueError("candidates must contain ProspectiveV2CandidateV1 values")
        kinds = tuple(candidate.candidate_kind for candidate in candidates)
        if kinds != PROSPECTIVE_V2_CANDIDATE_KINDS:
            raise ValueError(
                "candidate ladder must exactly match the registered V2 comparison"
            )
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        artifact_ids = tuple(candidate.artifact_id for candidate in candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("candidate artifact IDs must be unique")
        if any(candidate.target_outcomes_used for candidate in candidates):
            raise ValueError("promotion candidates must remain target-free")

        units = tuple(self.evaluation_units)
        if not units or any(
            type(value) is not ProspectiveV2EvaluationUnitV1 for value in units
        ):
            raise ValueError(
                "evaluation_units must contain ProspectiveV2EvaluationUnitV1 values"
            )
        unit_ids = tuple(unit.unit_id for unit in units)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("evaluation unit IDs must be unique")
        independence_keys = tuple(
            (unit.endpoint, unit.independent_group_id) for unit in units
        )
        if len(set(independence_keys)) != len(independence_keys):
            raise ValueError(
                "independent_group_id must be unique within each endpoint"
            )
        for endpoint in DECISION_TRACE_ENDPOINTS:
            count = sum(unit.endpoint == endpoint for unit in units)
            if count < self.policy.minimum_units_per_endpoint:
                raise ValueError(
                    f"endpoint {endpoint!r} has fewer registered independent units "
                    "than required by policy"
                )
        source_ids = _validated_string_tuple(
            self.source_artifact_ids,
            name="source_artifact_ids",
            require_sha256=True,
        )
        target_outcomes_used = _require_no_target_access(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        metadata = _validated_source_metadata(
            self.metadata,
            name="promotion-freeze metadata",
        )
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "stack_lock_id", stack_lock_id)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "evaluation_units", units)
        object.__setattr__(self, "source_artifact_ids", source_ids)
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(self, "metadata", metadata)

    @property
    def freeze_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
                "artifact_kind": "Causal4DProspectiveV2PromotionFreezeV1",
                "experiment_id": self.experiment_id,
                "stack_lock_id": self.stack_lock_id,
                "candidates": [candidate.as_dict() for candidate in self.candidates],
                "evaluation_units": [
                    unit.as_dict() for unit in self.evaluation_units
                ],
                "policy": asdict(self.policy),
                "source_artifact_ids": list(self.source_artifact_ids),
                "target_outcomes_used": self.target_outcomes_used,
                "metadata": plain_json(self.metadata),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2PromotionFreezeV1",
            "freeze_id": self.freeze_id,
            "experiment_id": self.experiment_id,
            "stack_lock_id": self.stack_lock_id,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "evaluation_units": [unit.as_dict() for unit in self.evaluation_units],
            "policy": asdict(self.policy),
            "source_artifact_ids": list(self.source_artifact_ids),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }
