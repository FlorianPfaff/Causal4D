"""Versioned deployment-decision profile for the prospective Causal4D V2 path.

The generic decision trace deliberately permits experiment-specific decision
inventories. This module freezes the stricter inventory needed by the
prospective V2 stack so a candidate cannot be selected while silently omitting
joint-covariance, functional-support, or conditional-uncertainty evidence.
It is additive and does not change the frozen physical-acquisition estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence, TypeVar

from causal4d.decision_trace import (
    DecisionTraceArtifact,
    DecisionTraceBuildResult,
    DecisionTraceDecision,
    DecisionTraceStage,
    TraceEndpoint,
    UnifiedDecisionTrace,
    build_unified_decision_trace,
)
from causal4d.immutable_json import plain_json


PROSPECTIVE_V2_PROFILE_SCHEMA_VERSION = 1
PROSPECTIVE_V2_PROFILE_ID = "causal4d.prospective-v2-deployment-profile/v1"
PROSPECTIVE_V2_REQUIRED_DECISION_NAMES = (
    "prob4d_provider_acceptance",
    "joint_covariance_admission",
    "bayesian_phystwin_acceptance",
    "functional_support",
    "intervention_identifiability",
    "action_support",
    "contact_v2_support",
    "conditional_uncertainty_calibration",
    "query_calibration",
    "counterfactual_regret",
)

_EXPECTED_DECISION_LOCATION = {
    "prob4d_provider_acceptance": ("prob4d_observation", "prob4d"),
    "joint_covariance_admission": ("causal4d_abduction", "causal4d"),
    "bayesian_phystwin_acceptance": (
        "bayesian_phystwin_belief",
        "bayesian-phystwin",
    ),
    "functional_support": ("causal4d_abduction", "causal4d"),
    "intervention_identifiability": ("causal4d_abduction", "causal4d"),
    "action_support": ("causal4d_abduction", "causal4d"),
    "contact_v2_support": ("causal4d_abduction", "causal4d"),
    "conditional_uncertainty_calibration": (
        "causal4d_counterfactual",
        "causal4d",
    ),
    "query_calibration": ("causal4d_counterfactual", "causal4d"),
    "counterfactual_regret": ("deployment", "causal4d"),
}

BaselineT = TypeVar("BaselineT")
CandidateT = TypeVar("CandidateT")


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


def _require_sha256(value: Any, *, name: str) -> str:
    digest = _require_nonempty_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class ProspectiveV2ProfileValidationV1:
    """Content-addressed result of applying the V2 trace profile."""

    trace_id: str
    accepted: bool
    reasons: tuple[str, ...]
    required_decision_names: tuple[str, ...]
    rejected_required_decision_names: tuple[str, ...]
    candidate_selected: bool

    def __post_init__(self) -> None:
        trace_id = _require_sha256(self.trace_id, name="trace_id")
        accepted = _require_bool(self.accepted, name="accepted")
        candidate_selected = _require_bool(
            self.candidate_selected,
            name="candidate_selected",
        )
        reasons = tuple(self.reasons)
        required = tuple(self.required_decision_names)
        rejected = tuple(self.rejected_required_decision_names)
        if required != PROSPECTIVE_V2_REQUIRED_DECISION_NAMES:
            raise ValueError("required_decision_names must equal the V2 profile")
        if len(set(reasons)) != len(reasons):
            raise ValueError("profile validation reasons must be unique")
        if len(set(rejected)) != len(rejected):
            raise ValueError("rejected required decisions must be unique")
        if any(name not in required for name in rejected):
            raise ValueError("rejected decisions must belong to the V2 profile")
        expected_accepted = not reasons
        if accepted is not expected_accepted:
            raise ValueError("profile validation decision must match its reasons")
        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "candidate_selected", candidate_selected)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "required_decision_names", required)
        object.__setattr__(self, "rejected_required_decision_names", rejected)

    @property
    def validation_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": PROSPECTIVE_V2_PROFILE_SCHEMA_VERSION,
                "artifact_kind": "Causal4DProspectiveV2ProfileValidationV1",
                "profile_id": PROSPECTIVE_V2_PROFILE_ID,
                "trace_id": self.trace_id,
                "accepted": self.accepted,
                "reasons": list(self.reasons),
                "required_decision_names": list(self.required_decision_names),
                "rejected_required_decision_names": list(
                    self.rejected_required_decision_names
                ),
                "candidate_selected": self.candidate_selected,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_PROFILE_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2ProfileValidationV1",
            "profile_id": PROSPECTIVE_V2_PROFILE_ID,
            "validation_id": self.validation_id,
            "trace_id": self.trace_id,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "required_decision_names": list(self.required_decision_names),
            "rejected_required_decision_names": list(
                self.rejected_required_decision_names
            ),
            "candidate_selected": self.candidate_selected,
        }


def _decision_index(
    stages: Sequence[DecisionTraceStage],
) -> tuple[
    dict[str, list[tuple[DecisionTraceStage, DecisionTraceDecision]]],
    list[str],
]:
    index: dict[
        str, list[tuple[DecisionTraceStage, DecisionTraceDecision]]
    ] = {}
    duplicate_names: list[str] = []
    for stage in stages:
        for decision in stage.decisions:
            values = index.setdefault(decision.name, [])
            values.append((stage, decision))
            if len(values) == 2:
                duplicate_names.append(decision.name)
    return index, duplicate_names


def validate_prospective_v2_decision_trace_v1(
    trace: UnifiedDecisionTrace,
) -> ProspectiveV2ProfileValidationV1:
    """Validate that a generic trace satisfies the complete V2 profile."""

    if type(trace) is not UnifiedDecisionTrace:
        raise ValueError("trace must be a UnifiedDecisionTrace")
    reasons: list[str] = []
    profile_id = trace.metadata.get("decision_profile_id")
    if profile_id != PROSPECTIVE_V2_PROFILE_ID:
        reasons.append("decision_profile_id_mismatch")

    index, duplicate_names = _decision_index(trace.stages)
    for name in duplicate_names:
        reasons.append(f"duplicate_required_decision:{name}")
    rejected: list[str] = []
    observed_names: list[str] = []
    for name in PROSPECTIVE_V2_REQUIRED_DECISION_NAMES:
        entries = index.get(name, [])
        if not entries:
            reasons.append(f"missing_required_decision:{name}")
            continue
        if len(entries) != 1:
            continue
        stage, decision = entries[0]
        observed_names.append(name)
        expected_stage, expected_producer = _EXPECTED_DECISION_LOCATION[name]
        if stage.stage_kind != expected_stage:
            reasons.append(
                f"required_decision_wrong_stage:{name}:{stage.stage_kind}"
            )
        if decision.producer != expected_producer:
            reasons.append(
                f"required_decision_wrong_producer:{name}:{decision.producer}"
            )
        if not decision.accepted:
            rejected.append(name)

    if trace.selection.required_decision_names != (
        PROSPECTIVE_V2_REQUIRED_DECISION_NAMES
    ):
        reasons.append("selection_required_decision_inventory_mismatch")
    expected_candidate_selected = not rejected and len(observed_names) == len(
        PROSPECTIVE_V2_REQUIRED_DECISION_NAMES
    )
    if (
        not reasons
        and trace.selection.candidate_selected != expected_candidate_selected
    ):
        reasons.append("selection_state_disagrees_with_v2_profile")

    return ProspectiveV2ProfileValidationV1(
        trace_id=trace.trace_id,
        accepted=not reasons,
        reasons=tuple(reasons),
        required_decision_names=PROSPECTIVE_V2_REQUIRED_DECISION_NAMES,
        rejected_required_decision_names=tuple(rejected),
        candidate_selected=trace.selection.candidate_selected,
    )


def require_prospective_v2_decision_trace_v1(
    trace: UnifiedDecisionTrace,
) -> UnifiedDecisionTrace:
    """Return a trace only when it satisfies the complete V2 profile."""

    validation = validate_prospective_v2_decision_trace_v1(trace)
    if not validation.accepted:
        joined = ", ".join(validation.reasons)
        raise ValueError(f"decision trace does not satisfy the V2 profile: {joined}")
    return trace


def build_prospective_v2_decision_trace_v1(
    *,
    trace_name: str,
    protocol_id: str,
    case_id: str,
    session_id: str,
    endpoint: TraceEndpoint,
    stack_lock_id: str,
    root_artifacts: Sequence[DecisionTraceArtifact],
    stages: Sequence[DecisionTraceStage],
    baseline: BaselineT,
    candidate: CandidateT,
    deployed: BaselineT | CandidateT,
    baseline_artifact_id: str,
    candidate_artifact_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> DecisionTraceBuildResult[BaselineT, CandidateT]:
    """Build a generic trace with the exact prospective V2 inventory."""

    trace_metadata = {} if metadata is None else dict(metadata)
    declared_profile = trace_metadata.get("decision_profile_id")
    if declared_profile not in (None, PROSPECTIVE_V2_PROFILE_ID):
        raise ValueError("metadata declares a different decision profile")
    trace_metadata["decision_profile_id"] = PROSPECTIVE_V2_PROFILE_ID
    result = build_unified_decision_trace(
        trace_name=trace_name,
        protocol_id=protocol_id,
        case_id=case_id,
        session_id=session_id,
        endpoint=endpoint,
        stack_lock_id=stack_lock_id,
        root_artifacts=root_artifacts,
        stages=stages,
        required_decision_names=PROSPECTIVE_V2_REQUIRED_DECISION_NAMES,
        baseline=baseline,
        candidate=candidate,
        deployed=deployed,
        baseline_artifact_id=baseline_artifact_id,
        candidate_artifact_id=candidate_artifact_id,
        metadata=trace_metadata,
    )
    require_prospective_v2_decision_trace_v1(result.trace)
    return result


__all__ = [
    "PROSPECTIVE_V2_PROFILE_ID",
    "PROSPECTIVE_V2_PROFILE_SCHEMA_VERSION",
    "PROSPECTIVE_V2_REQUIRED_DECISION_NAMES",
    "ProspectiveV2ProfileValidationV1",
    "build_prospective_v2_decision_trace_v1",
    "require_prospective_v2_decision_trace_v1",
    "validate_prospective_v2_decision_trace_v1",
]
