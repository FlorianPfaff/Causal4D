"""Content-addressed Prob4D -> BayesianPhysTwin -> Causal4D decision traces.

The trace is an additive deployment/audit contract.  It records artifact and
decision identities, not model payloads or target outcomes.  The fixed stage
order makes the causal and software hand-offs explicit while preserving exact
fallback semantics at the runtime construction boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Generic, Literal, Mapping, Sequence, TypeVar, cast

from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.atomic_io import atomic_write_json
from causal4d.immutable_json import plain_json, validated_json_mapping


DECISION_TRACE_SCHEMA_NAME = "causal4d.decision-trace"
DECISION_TRACE_SCHEMA_VERSION = 1
DECISION_TRACE_PIPELINE = ("prob4d", "bayesian-phystwin", "causal4d")
DECISION_TRACE_ENDPOINTS = (
    "factual_continuation",
    "same_grasp_transfer",
    "new_contact_transfer",
)
DECISION_TRACE_STAGE_KINDS = (
    "prob4d_observation",
    "bayesian_phystwin_belief",
    "causal4d_abduction",
    "causal4d_counterfactual",
    "deployment",
)

TraceProducer = Literal["prob4d", "bayesian-phystwin", "causal4d"]
TraceEndpoint = Literal[
    "factual_continuation",
    "same_grasp_transfer",
    "new_contact_transfer",
]
TraceStageKind = Literal[
    "prob4d_observation",
    "bayesian_phystwin_belief",
    "causal4d_abduction",
    "causal4d_counterfactual",
    "deployment",
]

BaselineT = TypeVar("BaselineT")
CandidateT = TypeVar("CandidateT")

_STAGE_PRODUCER: dict[str, str] = {
    "prob4d_observation": "prob4d",
    "bayesian_phystwin_belief": "bayesian-phystwin",
    "causal4d_abduction": "causal4d",
    "causal4d_counterfactual": "causal4d",
    "deployment": "causal4d",
}
_REQUIRED_ROOT_ROLES = {
    "factual_evidence_context",
    "counterfactual_query_context",
}
_REQUIRED_ROOT_PRODUCER = {
    "factual_evidence_context": "causal4d",
    "counterfactual_query_context": "causal4d",
}
_REQUIRED_OUTPUT_ROLES: dict[str, set[str]] = {
    "prob4d_observation": {"prob4d_observation"},
    "bayesian_phystwin_belief": {
        "bayesian_phystwin_belief",
        "baseline_prediction",
    },
    "causal4d_abduction": {"causal4d_factual_posterior"},
    "causal4d_counterfactual": {"candidate_prediction"},
    "deployment": set(),
}
_REQUIRED_INPUT_ROLES: dict[str, set[str]] = {
    "prob4d_observation": {"factual_evidence_context"},
    "bayesian_phystwin_belief": {"prob4d_observation"},
    "causal4d_abduction": {
        "prob4d_observation",
        "bayesian_phystwin_belief",
    },
    "causal4d_counterfactual": {
        "causal4d_factual_posterior",
        "counterfactual_query_context",
    },
    "deployment": {"baseline_prediction", "candidate_prediction"},
}
_FORBIDDEN_TARGET_METADATA_KEYS = {
    "evaluation_target",
    "held_out_target",
    "target_continuation",
    "target_future",
    "target_loss",
    "target_outcome",
    "target_outcomes",
}


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(kind: str, payload: Mapping[str, Any]) -> str:
    descriptor = {
        "schema_name": DECISION_TRACE_SCHEMA_NAME,
        "schema_version": DECISION_TRACE_SCHEMA_VERSION,
        "kind": kind,
        "payload": plain_json(payload),
    }
    return hashlib.sha256(_canonical_bytes(descriptor)).hexdigest()


def _require_exact_fields(
    values: Mapping[str, Any],
    *,
    fields: set[str],
    name: str,
) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in values):
        raise ValueError(f"{name} keys must be strings")
    actual = set(values)
    missing = sorted(fields - actual)
    unexpected = sorted(actual - fields)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match the schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


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
        raise ValueError(f"{name} must be a Boolean")
    return value


def _require_string_sequence(
    values: Any,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not allow_empty and not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _reject_target_future_metadata(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_TARGET_METADATA_KEYS:
                raise ValueError(
                    f"{path}.{key} is forbidden at the target-safe decision boundary"
                )
            _reject_target_future_metadata(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_target_future_metadata(item, path=f"{path}[{index}]")


def _validated_metadata(values: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    metadata = validated_json_mapping(
        values,
        error_message=f"{name} must contain finite JSON data",
    )
    _reject_target_future_metadata(metadata, path=name)
    return metadata


@dataclass(frozen=True)
class DecisionTraceArtifact:
    """Reference one immutable artifact without embedding its payload."""

    artifact_id: str
    artifact_kind: str
    role: str
    producer: TraceProducer
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        artifact_id = _require_sha256(self.artifact_id, name="artifact_id")
        artifact_kind = _require_nonempty_string(
            self.artifact_kind,
            name="artifact_kind",
        )
        role = _require_nonempty_string(self.role, name="role")
        producer = _require_nonempty_string(self.producer, name="producer")
        if producer not in DECISION_TRACE_PIPELINE:
            raise ValueError(f"producer must be one of {list(DECISION_TRACE_PIPELINE)}")
        if role in {"evaluation_target", "held_out_target"}:
            raise ValueError(
                "held-out target artifacts are forbidden in decision traces"
            )
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "artifact_kind", artifact_kind)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(
            self,
            "metadata",
            _validated_metadata(self.metadata, name="artifact metadata"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "role": self.role,
            "producer": self.producer,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DecisionTraceArtifact:
        _require_exact_fields(
            values,
            fields={
                "artifact_id",
                "artifact_kind",
                "role",
                "producer",
                "metadata",
            },
            name="decision-trace artifact",
        )
        producer = _require_nonempty_string(
            values["producer"],
            name="decision-trace artifact.producer",
        )
        if producer not in DECISION_TRACE_PIPELINE:
            raise ValueError("decision-trace artifact producer is invalid")
        return cls(
            artifact_id=_require_sha256(
                values["artifact_id"],
                name="decision-trace artifact.artifact_id",
            ),
            artifact_kind=_require_nonempty_string(
                values["artifact_kind"],
                name="decision-trace artifact.artifact_kind",
            ),
            role=_require_nonempty_string(
                values["role"],
                name="decision-trace artifact.role",
            ),
            producer=cast(TraceProducer, producer),
            metadata=_require_mapping(
                values["metadata"],
                name="decision-trace artifact.metadata",
            ),
        )


@dataclass(frozen=True)
class DecisionTraceDecision:
    """Reference one immutable admission/abstention decision."""

    name: str
    decision_id: str
    decision_kind: str
    producer: TraceProducer
    accepted: bool
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _require_nonempty_string(self.name, name="decision name")
        decision_id = _require_sha256(self.decision_id, name="decision_id")
        decision_kind = _require_nonempty_string(
            self.decision_kind,
            name="decision_kind",
        )
        producer = _require_nonempty_string(self.producer, name="producer")
        if producer not in DECISION_TRACE_PIPELINE:
            raise ValueError(f"producer must be one of {list(DECISION_TRACE_PIPELINE)}")
        accepted = _require_bool(self.accepted, name="accepted")
        reasons = _require_string_sequence(
            self.reasons,
            name="reasons",
            allow_empty=True,
        )
        if accepted and reasons:
            raise ValueError("accepted decisions cannot contain rejection reasons")
        if not accepted and not reasons:
            raise ValueError("rejected decisions require at least one reason")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "decision_id", decision_id)
        object.__setattr__(self, "decision_kind", decision_kind)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "metadata",
            _validated_metadata(self.metadata, name="decision metadata"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decision_id": self.decision_id,
            "decision_kind": self.decision_kind,
            "producer": self.producer,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DecisionTraceDecision:
        _require_exact_fields(
            values,
            fields={
                "name",
                "decision_id",
                "decision_kind",
                "producer",
                "accepted",
                "reasons",
                "metadata",
            },
            name="decision-trace decision",
        )
        producer = _require_nonempty_string(
            values["producer"],
            name="decision-trace decision.producer",
        )
        if producer not in DECISION_TRACE_PIPELINE:
            raise ValueError("decision-trace decision producer is invalid")
        return cls(
            name=_require_nonempty_string(
                values["name"],
                name="decision-trace decision.name",
            ),
            decision_id=_require_sha256(
                values["decision_id"],
                name="decision-trace decision.decision_id",
            ),
            decision_kind=_require_nonempty_string(
                values["decision_kind"],
                name="decision-trace decision.decision_kind",
            ),
            producer=cast(TraceProducer, producer),
            accepted=_require_bool(
                values["accepted"],
                name="decision-trace decision.accepted",
            ),
            reasons=_require_string_sequence(
                _require_list(
                    values["reasons"],
                    name="decision-trace decision.reasons",
                ),
                name="decision-trace decision.reasons",
                allow_empty=True,
            ),
            metadata=_require_mapping(
                values["metadata"],
                name="decision-trace decision.metadata",
            ),
        )


@dataclass(frozen=True)
class DecisionTraceStage:
    """One topologically ordered hand-off in the three-repository stack."""

    stage_name: str
    stage_kind: TraceStageKind
    producer: TraceProducer
    input_artifact_ids: tuple[str, ...]
    output_artifacts: tuple[DecisionTraceArtifact, ...] = ()
    decisions: tuple[DecisionTraceDecision, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stage_name = _require_nonempty_string(self.stage_name, name="stage_name")
        stage_kind = _require_nonempty_string(self.stage_kind, name="stage_kind")
        if stage_kind not in DECISION_TRACE_STAGE_KINDS:
            raise ValueError(
                f"stage_kind must be one of {list(DECISION_TRACE_STAGE_KINDS)}"
            )
        producer = _require_nonempty_string(self.producer, name="producer")
        expected_producer = _STAGE_PRODUCER[stage_kind]
        if producer != expected_producer:
            raise ValueError(f"{stage_kind} must be produced by {expected_producer}")
        inputs = tuple(
            _require_sha256(value, name=f"input_artifact_ids[{index}]")
            for index, value in enumerate(self.input_artifact_ids)
        )
        if not inputs or len(set(inputs)) != len(inputs):
            raise ValueError("input_artifact_ids must be nonempty and unique")
        artifacts = tuple(self.output_artifacts)
        decisions = tuple(self.decisions)
        if stage_kind != "deployment" and not artifacts:
            raise ValueError(f"{stage_kind} must publish at least one artifact")
        if stage_kind == "deployment" and artifacts:
            raise ValueError(
                "deployment selects an existing artifact and has no outputs"
            )
        if any(type(value) is not DecisionTraceArtifact for value in artifacts):
            raise ValueError(
                "output_artifacts must contain DecisionTraceArtifact values"
            )
        if any(type(value) is not DecisionTraceDecision for value in decisions):
            raise ValueError("decisions must contain DecisionTraceDecision values")
        artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("stage output artifact IDs must be unique")
        artifact_roles = tuple(artifact.role for artifact in artifacts)
        if len(set(artifact_roles)) != len(artifact_roles):
            raise ValueError("stage output artifact roles must be unique")
        if set(inputs) & set(artifact_ids):
            raise ValueError("a stage cannot consume and publish the same artifact ID")
        if any(artifact.producer != producer for artifact in artifacts):
            raise ValueError("stage outputs must identify the stage producer")
        decision_names = tuple(decision.name for decision in decisions)
        decision_ids = tuple(decision.decision_id for decision in decisions)
        if len(set(decision_names)) != len(decision_names):
            raise ValueError("stage decision names must be unique")
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("stage decision IDs must be unique")
        if any(decision.producer != producer for decision in decisions):
            raise ValueError("stage decisions must identify the stage producer")
        object.__setattr__(self, "stage_name", stage_name)
        object.__setattr__(self, "stage_kind", stage_kind)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "input_artifact_ids", inputs)
        object.__setattr__(self, "output_artifacts", artifacts)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "metadata",
            _validated_metadata(self.metadata, name="stage metadata"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "stage_kind": self.stage_kind,
            "producer": self.producer,
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifacts": [
                artifact.as_dict() for artifact in self.output_artifacts
            ],
            "decisions": [decision.as_dict() for decision in self.decisions],
            "metadata": plain_json(self.metadata),
        }

    @property
    def stage_id(self) -> str:
        return _content_id("DecisionTraceStage", self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "stage_id": self.stage_id}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DecisionTraceStage:
        _require_exact_fields(
            values,
            fields={
                "stage_id",
                "stage_name",
                "stage_kind",
                "producer",
                "input_artifact_ids",
                "output_artifacts",
                "decisions",
                "metadata",
            },
            name="decision-trace stage",
        )
        stage_kind = _require_nonempty_string(
            values["stage_kind"],
            name="decision-trace stage.stage_kind",
        )
        if stage_kind not in DECISION_TRACE_STAGE_KINDS:
            raise ValueError("decision-trace stage kind is invalid")
        producer = _require_nonempty_string(
            values["producer"],
            name="decision-trace stage.producer",
        )
        if producer not in DECISION_TRACE_PIPELINE:
            raise ValueError("decision-trace stage producer is invalid")
        stage = cls(
            stage_name=_require_nonempty_string(
                values["stage_name"],
                name="decision-trace stage.stage_name",
            ),
            stage_kind=cast(TraceStageKind, stage_kind),
            producer=cast(TraceProducer, producer),
            input_artifact_ids=tuple(
                _require_sha256(
                    value,
                    name=f"decision-trace stage.input_artifact_ids[{index}]",
                )
                for index, value in enumerate(
                    _require_list(
                        values["input_artifact_ids"],
                        name="decision-trace stage.input_artifact_ids",
                    )
                )
            ),
            output_artifacts=tuple(
                DecisionTraceArtifact.from_dict(
                    _require_mapping(
                        value,
                        name=f"decision-trace stage.output_artifacts[{index}]",
                    )
                )
                for index, value in enumerate(
                    _require_list(
                        values["output_artifacts"],
                        name="decision-trace stage.output_artifacts",
                    )
                )
            ),
            decisions=tuple(
                DecisionTraceDecision.from_dict(
                    _require_mapping(
                        value,
                        name=f"decision-trace stage.decisions[{index}]",
                    )
                )
                for index, value in enumerate(
                    _require_list(
                        values["decisions"],
                        name="decision-trace stage.decisions",
                    )
                )
            ),
            metadata=_require_mapping(
                values["metadata"],
                name="decision-trace stage.metadata",
            ),
        )
        expected = _require_sha256(
            values["stage_id"],
            name="decision-trace stage.stage_id",
        )
        if stage.stage_id != expected:
            raise ValueError("decision-trace stage_id does not match its payload")
        return stage


@dataclass(frozen=True)
class DecisionTraceSelection:
    """Bind the final deployment choice and the required decision inventory."""

    baseline_artifact_id: str
    candidate_artifact_id: str
    deployed_artifact_id: str
    candidate_selected: bool
    exact_object_identity_verified: bool
    required_decision_names: tuple[str, ...]

    def __post_init__(self) -> None:
        baseline_id = _require_sha256(
            self.baseline_artifact_id,
            name="baseline_artifact_id",
        )
        candidate_id = _require_sha256(
            self.candidate_artifact_id,
            name="candidate_artifact_id",
        )
        deployed_id = _require_sha256(
            self.deployed_artifact_id,
            name="deployed_artifact_id",
        )
        if baseline_id == candidate_id:
            raise ValueError("baseline and candidate artifact IDs must differ")
        selected = _require_bool(
            self.candidate_selected,
            name="candidate_selected",
        )
        identity_verified = _require_bool(
            self.exact_object_identity_verified,
            name="exact_object_identity_verified",
        )
        if not identity_verified:
            raise ValueError("exact runtime selection identity must be verified")
        expected = candidate_id if selected else baseline_id
        if deployed_id != expected:
            raise ValueError("deployed_artifact_id disagrees with candidate_selected")
        required = _require_string_sequence(
            self.required_decision_names,
            name="required_decision_names",
            allow_empty=False,
        )
        object.__setattr__(self, "baseline_artifact_id", baseline_id)
        object.__setattr__(self, "candidate_artifact_id", candidate_id)
        object.__setattr__(self, "deployed_artifact_id", deployed_id)
        object.__setattr__(self, "candidate_selected", selected)
        object.__setattr__(self, "exact_object_identity_verified", identity_verified)
        object.__setattr__(self, "required_decision_names", required)

    def _payload(self) -> dict[str, Any]:
        return {
            "baseline_artifact_id": self.baseline_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "deployed_artifact_id": self.deployed_artifact_id,
            "candidate_selected": self.candidate_selected,
            "exact_object_identity_verified": self.exact_object_identity_verified,
            "required_decision_names": list(self.required_decision_names),
        }

    @property
    def selection_id(self) -> str:
        return _content_id("DecisionTraceSelection", self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "selection_id": self.selection_id}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DecisionTraceSelection:
        _require_exact_fields(
            values,
            fields={
                "selection_id",
                "baseline_artifact_id",
                "candidate_artifact_id",
                "deployed_artifact_id",
                "candidate_selected",
                "exact_object_identity_verified",
                "required_decision_names",
            },
            name="decision-trace selection",
        )
        selection = cls(
            baseline_artifact_id=_require_sha256(
                values["baseline_artifact_id"],
                name="decision-trace selection.baseline_artifact_id",
            ),
            candidate_artifact_id=_require_sha256(
                values["candidate_artifact_id"],
                name="decision-trace selection.candidate_artifact_id",
            ),
            deployed_artifact_id=_require_sha256(
                values["deployed_artifact_id"],
                name="decision-trace selection.deployed_artifact_id",
            ),
            candidate_selected=_require_bool(
                values["candidate_selected"],
                name="decision-trace selection.candidate_selected",
            ),
            exact_object_identity_verified=_require_bool(
                values["exact_object_identity_verified"],
                name="decision-trace selection.exact_object_identity_verified",
            ),
            required_decision_names=_require_string_sequence(
                _require_list(
                    values["required_decision_names"],
                    name="decision-trace selection.required_decision_names",
                ),
                name="decision-trace selection.required_decision_names",
                allow_empty=False,
            ),
        )
        expected = _require_sha256(
            values["selection_id"],
            name="decision-trace selection.selection_id",
        )
        if selection.selection_id != expected:
            raise ValueError("decision-trace selection_id does not match its payload")
        return selection


@dataclass(frozen=True)
class UnifiedDecisionTrace:
    """A target-safe, content-addressed decision DAG for the full stack."""

    trace_name: str
    protocol_id: str
    case_id: str
    session_id: str
    endpoint: TraceEndpoint
    stack_lock_id: str
    root_artifacts: tuple[DecisionTraceArtifact, ...]
    stages: tuple[DecisionTraceStage, ...]
    selection: DecisionTraceSelection
    target_future_observations_read: int = 0
    target_future_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        trace_name = _require_nonempty_string(self.trace_name, name="trace_name")
        protocol_id = _require_nonempty_string(self.protocol_id, name="protocol_id")
        case_id = _require_nonempty_string(self.case_id, name="case_id")
        session_id = _require_nonempty_string(self.session_id, name="session_id")
        endpoint = _require_nonempty_string(self.endpoint, name="endpoint")
        if endpoint not in DECISION_TRACE_ENDPOINTS:
            raise ValueError(
                f"endpoint must be one of {list(DECISION_TRACE_ENDPOINTS)}"
            )
        stack_lock_id = _require_sha256(self.stack_lock_id, name="stack_lock_id")
        roots = tuple(self.root_artifacts)
        stages = tuple(self.stages)
        if not roots or any(
            type(value) is not DecisionTraceArtifact for value in roots
        ):
            raise ValueError("root_artifacts must contain DecisionTraceArtifact values")
        if type(self.selection) is not DecisionTraceSelection:
            raise ValueError("selection must be a DecisionTraceSelection")
        if len(stages) != len(DECISION_TRACE_STAGE_KINDS) or any(
            type(value) is not DecisionTraceStage for value in stages
        ):
            raise ValueError(
                "stages must contain exactly the five decision-trace stages"
            )
        observed_stage_kinds = tuple(stage.stage_kind for stage in stages)
        if observed_stage_kinds != DECISION_TRACE_STAGE_KINDS:
            raise ValueError(
                "stages must follow the fixed Prob4D -> BPT -> Causal4D order"
            )
        stage_names = tuple(stage.stage_name for stage in stages)
        stage_ids = tuple(stage.stage_id for stage in stages)
        if len(set(stage_names)) != len(stage_names):
            raise ValueError("stage names must be unique")
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("stage IDs must be unique")
        if type(self.target_future_observations_read) is not int:
            raise ValueError("target_future_observations_read must be an integer")
        if self.target_future_observations_read != 0:
            raise ValueError("decision traces must not read target-future observations")
        if _require_bool(
            self.target_future_outcomes_used,
            name="target_future_outcomes_used",
        ):
            raise ValueError("decision traces must not use target-future outcomes")

        available: dict[str, DecisionTraceArtifact] = {}
        for artifact in roots:
            if artifact.artifact_id in available:
                raise ValueError("root artifact IDs must be unique")
            available[artifact.artifact_id] = artifact
        root_roles = {artifact.role for artifact in roots}
        missing_root_roles = sorted(_REQUIRED_ROOT_ROLES - root_roles)
        if missing_root_roles:
            raise ValueError(
                f"decision trace is missing root roles {missing_root_roles}"
            )
        for artifact in roots:
            expected_producer = _REQUIRED_ROOT_PRODUCER.get(artifact.role)
            if expected_producer is not None and artifact.producer != expected_producer:
                raise ValueError(
                    f"root role {artifact.role!r} must be produced by "
                    f"{expected_producer}"
                )

        decisions_by_name: dict[str, DecisionTraceDecision] = {}
        decision_ids: set[str] = set()
        for stage in stages:
            missing_inputs = sorted(
                artifact_id
                for artifact_id in stage.input_artifact_ids
                if artifact_id not in available
            )
            if missing_inputs:
                raise ValueError(
                    f"{stage.stage_name} consumes unavailable or forward artifacts "
                    f"{missing_inputs}"
                )
            input_roles = {
                available[artifact_id].role for artifact_id in stage.input_artifact_ids
            }
            required_inputs = _REQUIRED_INPUT_ROLES[stage.stage_kind]
            missing_input_roles = sorted(required_inputs - input_roles)
            if missing_input_roles:
                raise ValueError(
                    f"{stage.stage_kind} is missing input roles {missing_input_roles}"
                )
            output_roles = {artifact.role for artifact in stage.output_artifacts}
            required_outputs = _REQUIRED_OUTPUT_ROLES[stage.stage_kind]
            missing_output_roles = sorted(required_outputs - output_roles)
            if missing_output_roles:
                raise ValueError(
                    f"{stage.stage_kind} is missing output roles {missing_output_roles}"
                )
            for artifact in stage.output_artifacts:
                if artifact.artifact_id in available:
                    raise ValueError(
                        "artifact IDs must be globally unique across trace outputs"
                    )
                available[artifact.artifact_id] = artifact
            for decision in stage.decisions:
                if decision.name in decisions_by_name:
                    raise ValueError("decision names must be globally unique")
                if decision.decision_id in decision_ids:
                    raise ValueError("decision IDs must be globally unique")
                decisions_by_name[decision.name] = decision
                decision_ids.add(decision.decision_id)

        role_index: dict[str, list[DecisionTraceArtifact]] = {}
        for artifact in available.values():
            role_index.setdefault(artifact.role, []).append(artifact)
        for required_role in {
            *_REQUIRED_ROOT_ROLES,
            *set().union(*_REQUIRED_OUTPUT_ROLES.values()),
        }:
            if len(role_index.get(required_role, [])) != 1:
                raise ValueError(
                    f"decision trace requires exactly one artifact with role "
                    f"{required_role!r}"
                )

        deployment = stages[-1]
        baseline_role_id = role_index["baseline_prediction"][0].artifact_id
        candidate_role_id = role_index["candidate_prediction"][0].artifact_id
        if self.selection.baseline_artifact_id != baseline_role_id:
            raise ValueError("selection baseline does not match baseline_prediction")
        if self.selection.candidate_artifact_id != candidate_role_id:
            raise ValueError("selection candidate does not match candidate_prediction")
        if not {
            self.selection.baseline_artifact_id,
            self.selection.candidate_artifact_id,
        }.issubset(set(deployment.input_artifact_ids)):
            raise ValueError("deployment must consume both baseline and candidate")

        missing_decisions = sorted(
            set(self.selection.required_decision_names) - set(decisions_by_name)
        )
        if missing_decisions:
            raise ValueError(
                f"required decisions are absent from the trace: {missing_decisions}"
            )
        all_required_accepted = all(
            decisions_by_name[name].accepted
            for name in self.selection.required_decision_names
        )
        if self.selection.candidate_selected != all_required_accepted:
            raise ValueError(
                "candidate selection must equal the conjunction of required decisions"
            )

        object.__setattr__(self, "trace_name", trace_name)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "stack_lock_id", stack_lock_id)
        object.__setattr__(self, "root_artifacts", roots)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(
            self,
            "metadata",
            _validated_metadata(self.metadata, name="trace metadata"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "trace_name": self.trace_name,
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "session_id": self.session_id,
            "endpoint": self.endpoint,
            "stack_lock_id": self.stack_lock_id,
            "root_artifacts": [artifact.as_dict() for artifact in self.root_artifacts],
            "stages": [stage.as_dict() for stage in self.stages],
            "selection": self.selection.as_dict(),
            "target_future_observations_read": self.target_future_observations_read,
            "target_future_outcomes_used": self.target_future_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def trace_id(self) -> str:
        return _content_id("UnifiedDecisionTrace", self._payload())

    @property
    def decisions(self) -> tuple[DecisionTraceDecision, ...]:
        return tuple(decision for stage in self.stages for decision in stage.decisions)

    @property
    def artifacts(self) -> tuple[DecisionTraceArtifact, ...]:
        return (
            *self.root_artifacts,
            *tuple(
                artifact for stage in self.stages for artifact in stage.output_artifacts
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_name": DECISION_TRACE_SCHEMA_NAME,
            "schema_version": DECISION_TRACE_SCHEMA_VERSION,
            "trace_id": self.trace_id,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> UnifiedDecisionTrace:
        _require_exact_fields(
            values,
            fields={
                "schema_name",
                "schema_version",
                "trace_id",
                "trace_name",
                "protocol_id",
                "case_id",
                "session_id",
                "endpoint",
                "stack_lock_id",
                "root_artifacts",
                "stages",
                "selection",
                "target_future_observations_read",
                "target_future_outcomes_used",
                "metadata",
            },
            name="decision trace",
        )
        if values["schema_name"] != DECISION_TRACE_SCHEMA_NAME:
            raise ValueError("unsupported decision-trace schema name")
        if (
            type(values["schema_version"]) is not int
            or values["schema_version"] != DECISION_TRACE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported decision-trace schema version")
        endpoint = _require_nonempty_string(
            values["endpoint"],
            name="decision trace.endpoint",
        )
        if endpoint not in DECISION_TRACE_ENDPOINTS:
            raise ValueError("decision trace endpoint is invalid")
        observations_read = values["target_future_observations_read"]
        if type(observations_read) is not int:
            raise ValueError(
                "decision trace.target_future_observations_read must be an integer"
            )
        trace = cls(
            trace_name=_require_nonempty_string(
                values["trace_name"],
                name="decision trace.trace_name",
            ),
            protocol_id=_require_nonempty_string(
                values["protocol_id"],
                name="decision trace.protocol_id",
            ),
            case_id=_require_nonempty_string(
                values["case_id"],
                name="decision trace.case_id",
            ),
            session_id=_require_nonempty_string(
                values["session_id"],
                name="decision trace.session_id",
            ),
            endpoint=cast(TraceEndpoint, endpoint),
            stack_lock_id=_require_sha256(
                values["stack_lock_id"],
                name="decision trace.stack_lock_id",
            ),
            root_artifacts=tuple(
                DecisionTraceArtifact.from_dict(
                    _require_mapping(
                        value,
                        name=f"decision trace.root_artifacts[{index}]",
                    )
                )
                for index, value in enumerate(
                    _require_list(
                        values["root_artifacts"],
                        name="decision trace.root_artifacts",
                    )
                )
            ),
            stages=tuple(
                DecisionTraceStage.from_dict(
                    _require_mapping(
                        value,
                        name=f"decision trace.stages[{index}]",
                    )
                )
                for index, value in enumerate(
                    _require_list(
                        values["stages"],
                        name="decision trace.stages",
                    )
                )
            ),
            selection=DecisionTraceSelection.from_dict(
                _require_mapping(
                    values["selection"],
                    name="decision trace.selection",
                )
            ),
            target_future_observations_read=observations_read,
            target_future_outcomes_used=_require_bool(
                values["target_future_outcomes_used"],
                name="decision trace.target_future_outcomes_used",
            ),
            metadata=_require_mapping(
                values["metadata"],
                name="decision trace.metadata",
            ),
        )
        expected = _require_sha256(values["trace_id"], name="decision trace.trace_id")
        if trace.trace_id != expected:
            raise ValueError("decision trace_id does not match its payload")
        return trace


@dataclass(frozen=True)
class DecisionTraceBuildResult(Generic[BaselineT, CandidateT]):
    """Return the trace and the exact deployed runtime object together."""

    trace: UnifiedDecisionTrace
    baseline: BaselineT
    candidate: CandidateT
    deployed: BaselineT | CandidateT

    def __post_init__(self) -> None:
        expected = (
            self.candidate if self.trace.selection.candidate_selected else self.baseline
        )
        if self.deployed is not expected:
            raise ValueError("deployed object does not preserve exact trace selection")


def build_unified_decision_trace(
    *,
    trace_name: str,
    protocol_id: str,
    case_id: str,
    session_id: str,
    endpoint: TraceEndpoint,
    stack_lock_id: str,
    root_artifacts: Sequence[DecisionTraceArtifact],
    stages: Sequence[DecisionTraceStage],
    required_decision_names: Sequence[str],
    baseline: BaselineT,
    candidate: CandidateT,
    deployed: BaselineT | CandidateT,
    baseline_artifact_id: str,
    candidate_artifact_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> DecisionTraceBuildResult[BaselineT, CandidateT]:
    """Construct a trace while checking the exact runtime deployment identity."""

    if baseline is candidate:
        raise ValueError("baseline and candidate runtime objects must differ")
    decisions = {
        decision.name: decision for stage in stages for decision in stage.decisions
    }
    required = _require_string_sequence(
        required_decision_names,
        name="required_decision_names",
        allow_empty=False,
    )
    missing = sorted(set(required) - set(decisions))
    if missing:
        raise ValueError(f"required decisions are absent from stages: {missing}")
    candidate_selected = all(decisions[name].accepted for name in required)
    expected_object: BaselineT | CandidateT = (
        candidate if candidate_selected else baseline
    )
    if deployed is not expected_object:
        raise ValueError(
            "deployed object does not match the conjunction of required decisions"
        )
    selection = DecisionTraceSelection(
        baseline_artifact_id=baseline_artifact_id,
        candidate_artifact_id=candidate_artifact_id,
        deployed_artifact_id=(
            candidate_artifact_id if candidate_selected else baseline_artifact_id
        ),
        candidate_selected=candidate_selected,
        exact_object_identity_verified=True,
        required_decision_names=required,
    )
    trace = UnifiedDecisionTrace(
        trace_name=trace_name,
        protocol_id=protocol_id,
        case_id=case_id,
        session_id=session_id,
        endpoint=endpoint,
        stack_lock_id=stack_lock_id,
        root_artifacts=tuple(root_artifacts),
        stages=tuple(stages),
        selection=selection,
        metadata={} if metadata is None else metadata,
    )
    return DecisionTraceBuildResult(
        trace=trace,
        baseline=baseline,
        candidate=candidate,
        deployed=deployed,
    )


def write_decision_trace(
    path: str | Path,
    trace: UnifiedDecisionTrace,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one validated decision trace."""

    if type(trace) is not UnifiedDecisionTrace:
        raise ValueError("trace must be a UnifiedDecisionTrace")
    atomic_write_json(path, trace.as_dict(), overwrite=overwrite)


def load_decision_trace(path: str | Path) -> UnifiedDecisionTrace:
    """Load a duplicate-key-safe, finite, content-addressed decision trace."""

    snapshot = read_regular_file(path, name="decision trace")
    values = load_strict_json_object(snapshot.payload, name="decision trace")
    return UnifiedDecisionTrace.from_dict(values)


def load_claim_bearing_decision_trace(
    path: str | Path,
    *,
    expected_trace_id: str,
    expected_stack_lock_id: str,
    expected_protocol_id: str | None = None,
) -> UnifiedDecisionTrace:
    """Load a trace only when independently frozen identities match."""

    trace = load_decision_trace(path)
    trace_id = _require_sha256(expected_trace_id, name="expected_trace_id")
    stack_lock_id = _require_sha256(
        expected_stack_lock_id,
        name="expected_stack_lock_id",
    )
    if trace.trace_id != trace_id:
        raise ValueError("decision trace does not match expected_trace_id")
    if trace.stack_lock_id != stack_lock_id:
        raise ValueError("decision trace does not match expected_stack_lock_id")
    if expected_protocol_id is not None:
        protocol_id = _require_nonempty_string(
            expected_protocol_id,
            name="expected_protocol_id",
        )
        if trace.protocol_id != protocol_id:
            raise ValueError("decision trace does not match expected_protocol_id")
    return trace


def require_decision_trace_stack_lock(
    trace: UnifiedDecisionTrace,
    stack_lock: Mapping[str, Any],
) -> UnifiedDecisionTrace:
    """Bind a trace to a fully validated external three-repository stack lock."""

    from causal4d.stack_lock import validate_stack_lock

    validated = validate_stack_lock(stack_lock)
    if trace.stack_lock_id != validated["lock_id"]:
        raise ValueError("decision trace stack_lock_id does not match the stack lock")
    return trace


__all__ = [
    "DECISION_TRACE_ENDPOINTS",
    "DECISION_TRACE_PIPELINE",
    "DECISION_TRACE_SCHEMA_NAME",
    "DECISION_TRACE_SCHEMA_VERSION",
    "DECISION_TRACE_STAGE_KINDS",
    "DecisionTraceArtifact",
    "DecisionTraceBuildResult",
    "DecisionTraceDecision",
    "DecisionTraceSelection",
    "DecisionTraceStage",
    "UnifiedDecisionTrace",
    "build_unified_decision_trace",
    "load_claim_bearing_decision_trace",
    "load_decision_trace",
    "require_decision_trace_stack_lock",
    "write_decision_trace",
]
