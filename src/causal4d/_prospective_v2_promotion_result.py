"""One-opening selection result for prospective V2 promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from causal4d._prospective_v2_promotion_candidate_result import (
    ProspectiveV2CandidateResultV1,
)
from causal4d._prospective_v2_promotion_common import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
    _canonical_sha256,
    _require_bool,
    _require_nonempty_string,
    _require_sha256,
    _require_target_access,
    _validated_string_tuple,
)
from causal4d.immutable_json import plain_json, validated_json_mapping


@dataclass(frozen=True)
class ProspectiveV2PromotionResultV1:
    """Content-addressed one-opening result and exact artifact selection."""

    freeze_id: str
    evaluation_opening_id: str
    baseline_candidate_id: str
    baseline_artifact_id: str
    selected_candidate_id: str
    selected_candidate_kind: str
    selected_artifact_id: str
    candidate_results: tuple[ProspectiveV2CandidateResultV1, ...]
    evaluation_metric_ids: tuple[str, ...]
    exact_artifact_fallback_verified: bool
    one_target_opening_verified: bool
    target_outcomes_used: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        freeze_id = _require_sha256(self.freeze_id, name="freeze_id")
        opening_id = _require_sha256(
            self.evaluation_opening_id,
            name="evaluation_opening_id",
        )
        baseline_candidate_id = _require_nonempty_string(
            self.baseline_candidate_id,
            name="baseline_candidate_id",
        )
        baseline_artifact_id = _require_sha256(
            self.baseline_artifact_id,
            name="baseline_artifact_id",
        )
        selected_candidate_id = _require_nonempty_string(
            self.selected_candidate_id,
            name="selected_candidate_id",
        )
        selected_kind = _require_nonempty_string(
            self.selected_candidate_kind,
            name="selected_candidate_kind",
        )
        if selected_kind not in PROSPECTIVE_V2_CANDIDATE_KINDS:
            raise ValueError("selected_candidate_kind is invalid")
        selected_artifact_id = _require_sha256(
            self.selected_artifact_id,
            name="selected_artifact_id",
        )
        candidate_results = tuple(self.candidate_results)
        if any(
            type(result) is not ProspectiveV2CandidateResultV1
            for result in candidate_results
        ):
            raise ValueError(
                "candidate_results must contain ProspectiveV2CandidateResultV1 values"
            )
        expected_kinds = PROSPECTIVE_V2_CANDIDATE_KINDS[1:]
        if tuple(result.candidate_kind for result in candidate_results) != (
            expected_kinds
        ):
            raise ValueError("candidate results must follow the frozen ladder")
        candidate_result_ids = tuple(
            result.candidate_id for result in candidate_results
        )
        candidate_result_artifacts = tuple(
            result.candidate_artifact_id for result in candidate_results
        )
        if len(set(candidate_result_ids)) != len(candidate_result_ids):
            raise ValueError("candidate result IDs must be unique")
        if len(set(candidate_result_artifacts)) != len(candidate_result_artifacts):
            raise ValueError("candidate result artifact IDs must be unique")
        if baseline_candidate_id in candidate_result_ids:
            raise ValueError("baseline candidate must not appear in candidate results")
        if baseline_artifact_id in candidate_result_artifacts:
            raise ValueError("baseline artifact must differ from candidate artifacts")
        accepted_results = tuple(
            result for result in candidate_results if result.accepted
        )
        if accepted_results:
            expected_selected = accepted_results[-1]
            expected_selected_fields = (
                expected_selected.candidate_id,
                expected_selected.candidate_kind,
                expected_selected.candidate_artifact_id,
            )
        else:
            expected_selected_fields = (
                baseline_candidate_id,
                PROSPECTIVE_V2_CANDIDATE_KINDS[0],
                baseline_artifact_id,
            )
        observed_selected_fields = (
            selected_candidate_id,
            selected_kind,
            selected_artifact_id,
        )
        if observed_selected_fields != expected_selected_fields:
            raise ValueError(
                "selected artifact must equal the highest accepted frozen candidate "
                "or the exact baseline"
            )
        metric_ids = _validated_string_tuple(
            self.evaluation_metric_ids,
            name="evaluation_metric_ids",
            require_sha256=True,
        )
        if not _require_bool(
            self.exact_artifact_fallback_verified,
            name="exact_artifact_fallback_verified",
        ):
            raise ValueError("exact artifact fallback must be verified")
        if not _require_bool(
            self.one_target_opening_verified,
            name="one_target_opening_verified",
        ):
            raise ValueError("one target opening must be verified")
        target_outcomes_used = _require_target_access(
            self.target_outcomes_used,
            name="target_outcomes_used",
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="promotion-result metadata must contain finite JSON data",
        )
        object.__setattr__(self, "freeze_id", freeze_id)
        object.__setattr__(self, "evaluation_opening_id", opening_id)
        object.__setattr__(self, "baseline_candidate_id", baseline_candidate_id)
        object.__setattr__(self, "baseline_artifact_id", baseline_artifact_id)
        object.__setattr__(self, "selected_candidate_id", selected_candidate_id)
        object.__setattr__(self, "selected_candidate_kind", selected_kind)
        object.__setattr__(self, "selected_artifact_id", selected_artifact_id)
        object.__setattr__(self, "candidate_results", candidate_results)
        object.__setattr__(self, "evaluation_metric_ids", metric_ids)
        object.__setattr__(self, "target_outcomes_used", target_outcomes_used)
        object.__setattr__(self, "metadata", metadata)

    @property
    def result_id(self) -> str:
        return _canonical_sha256(self.as_dict(include_result_id=False))

    def as_dict(self, *, include_result_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2PromotionResultV1",
            "freeze_id": self.freeze_id,
            "evaluation_opening_id": self.evaluation_opening_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "baseline_artifact_id": self.baseline_artifact_id,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_kind": self.selected_candidate_kind,
            "selected_artifact_id": self.selected_artifact_id,
            "candidate_results": [
                result.as_dict() for result in self.candidate_results
            ],
            "evaluation_metric_ids": list(self.evaluation_metric_ids),
            "exact_artifact_fallback_verified": (
                self.exact_artifact_fallback_verified
            ),
            "one_target_opening_verified": self.one_target_opening_verified,
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }
        if include_result_id:
            payload["result_id"] = self.result_id
        return payload
