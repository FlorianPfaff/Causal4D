"""Stage-specific provenance identities for causal inference and evaluation.

The version-1 :class:`~causal4d.contracts.CausalContext` intentionally remains
unchanged for frozen artifacts.  This module provides additive identities for a
future contract boundary in which factual evidence, the counterfactual query,
and the held-out evaluation target are content-addressed independently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Mapping, cast

import numpy as np

from causal4d.contracts import (
    ActionWindow,
    CausalContext,
    CounterfactualQuery,
    ObservationWindow,
    array_sha256,
)


STAGE_PROVENANCE_SCHEMA_VERSION = 1


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _artifact_id(contract_type: str, payload: Mapping[str, Any]) -> str:
    descriptor = {
        "schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
        "contract_type": contract_type,
        "payload": payload,
    }
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_fields(
    values: Mapping[str, Any],
    *,
    fields: set[str],
    name: str,
) -> None:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in values):
        raise ValueError(f"{name} keys must be strings")
    actual = set(values)
    missing = sorted(fields - actual)
    unexpected = sorted(actual - fields)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match the schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _require_string(values: Mapping[str, Any], key: str, *, name: str) -> str:
    value = values[key]
    if not isinstance(value, str):
        raise ValueError(f"{name}.{key} must be a string")
    return value


def _require_integer(values: Mapping[str, Any], key: str, *, name: str) -> int:
    value = values[key]
    if type(value) is not int:
        raise ValueError(f"{name}.{key} must be an integer")
    return value


def _require_mapping(
    values: Mapping[str, Any], key: str, *, name: str
) -> Mapping[str, Any]:
    value = values[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}.{key} must be a JSON object")
    return value


def _observation_window_from_dict(
    values: Mapping[str, Any], *, name: str
) -> ObservationWindow:
    _require_exact_fields(
        values,
        fields={
            "case_id",
            "stream_id",
            "frame_start",
            "frame_stop",
            "content_sha256",
        },
        name=name,
    )
    return ObservationWindow(
        case_id=_require_string(values, "case_id", name=name),
        stream_id=_require_string(values, "stream_id", name=name),
        frame_start=_require_integer(values, "frame_start", name=name),
        frame_stop=_require_integer(values, "frame_stop", name=name),
        content_sha256=_require_string(values, "content_sha256", name=name),
    )


def _action_window_from_dict(values: Mapping[str, Any], *, name: str) -> ActionWindow:
    _require_exact_fields(
        values,
        fields={
            "action_id",
            "case_id",
            "frame_start",
            "frame_stop",
            "trajectory_sha256",
            "provenance",
        },
        name=name,
    )
    return ActionWindow(
        action_id=_require_string(values, "action_id", name=name),
        case_id=_require_string(values, "case_id", name=name),
        frame_start=_require_integer(values, "frame_start", name=name),
        frame_stop=_require_integer(values, "frame_stop", name=name),
        trajectory_sha256=_require_string(
            values,
            "trajectory_sha256",
            name=name,
        ),
        provenance=_require_string(values, "provenance", name=name),
    )


def _require_header(
    values: Mapping[str, Any],
    *,
    contract_type: str,
    payload_fields: set[str],
) -> None:
    name = f"stage-provenance {contract_type}"
    _require_exact_fields(
        values,
        fields={"schema_version", "contract_type", *payload_fields},
        name=name,
    )
    if _require_integer(values, "schema_version", name=name) != (
        STAGE_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported stage-provenance schema version")
    if _require_string(values, "contract_type", name=name) != contract_type:
        raise ValueError(f"expected stage-provenance contract {contract_type}")


def _frame_array(values: np.ndarray, *, name: str, frame_stop: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1:
        raise ValueError(f"{name} must have a frame axis")
    if frame_stop > len(array):
        raise ValueError(f"{name} does not cover frame {frame_stop - 1}")
    return array


def _require_window_digest(
    values: np.ndarray,
    window: ObservationWindow | ActionWindow,
    *,
    name: str,
) -> None:
    array = _frame_array(values, name=name, frame_stop=window.frame_stop)
    digest = array_sha256(array[window.frame_start : window.frame_stop])
    expected = (
        window.content_sha256
        if isinstance(window, ObservationWindow)
        else window.trajectory_sha256
    )
    if digest != expected:
        raise ValueError(f"{name} does not match the declared {name} digest")


@dataclass(frozen=True)
class FactualEvidenceContext:
    """Content identity for exactly the evidence admitted by factual abduction.

    This context contains the pre-intervention observation window, the admitted
    post-intervention response prefix, and the known observed command.  It does
    not contain the counterfactual query or the held-out observation suffix.
    """

    contract_type: ClassVar[str] = "FactualEvidenceContext"

    protocol_id: str
    o_minus: ObservationWindow
    o_plus_prefix: ObservationWindow
    u_obs: ActionWindow

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must be nonempty")
        case_ids = {
            self.o_minus.case_id,
            self.o_plus_prefix.case_id,
            self.u_obs.case_id,
        }
        if len(case_ids) != 1:
            raise ValueError("factual evidence windows must identify the same case")
        if self.o_minus.frame_stop > self.o_plus_prefix.frame_start:
            raise ValueError("O- must not overlap the admitted O+ prefix")
        if (
            self.u_obs.frame_start > self.o_plus_prefix.frame_start
            or self.u_obs.frame_stop < self.o_plus_prefix.frame_stop
        ):
            raise ValueError("u_obs must cover the admitted O+ prefix")

    @property
    def case_id(self) -> str:
        return self.o_minus.case_id

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "o_minus": self.o_minus.as_dict(),
            "o_plus_prefix": self.o_plus_prefix.as_dict(),
            "u_obs": self.u_obs.as_dict(),
        }

    @property
    def artifact_id(self) -> str:
        return _artifact_id(self.contract_type, self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
            "contract_type": self.contract_type,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> FactualEvidenceContext:
        _require_header(
            values,
            contract_type=cls.contract_type,
            payload_fields={"protocol_id", "o_minus", "o_plus_prefix", "u_obs"},
        )
        name = f"stage-provenance {cls.contract_type}"
        return cls(
            protocol_id=_require_string(values, "protocol_id", name=name),
            o_minus=_observation_window_from_dict(
                _require_mapping(values, "o_minus", name=name),
                name=f"{name}.o_minus",
            ),
            o_plus_prefix=_observation_window_from_dict(
                _require_mapping(values, "o_plus_prefix", name=name),
                name=f"{name}.o_plus_prefix",
            ),
            u_obs=_action_window_from_dict(
                _require_mapping(values, "u_obs", name=name),
                name=f"{name}.u_obs",
            ),
        )


@dataclass(frozen=True)
class CounterfactualQueryContext:
    """Content identity for a ``do(u_cf)`` query independent of target outcomes."""

    contract_type: ClassVar[str] = "CounterfactualQueryContext"

    protocol_id: str
    case_id: str
    u_cf: ActionWindow
    contact_policy: Literal["same_grasp", "new_contact"]
    language: str | None = None
    query_node_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.protocol_id or not self.case_id:
            raise ValueError("protocol_id and case_id must be nonempty")
        if self.u_cf.case_id != self.case_id:
            raise ValueError("u_cf must identify the query case")
        if self.contact_policy not in {"same_grasp", "new_contact"}:
            raise ValueError("contact_policy must be 'same_grasp' or 'new_contact'")
        if self.query_node_indices is not None:
            if not self.query_node_indices or any(
                index < 0 for index in self.query_node_indices
            ):
                raise ValueError(
                    "query_node_indices must be a nonempty nonnegative tuple"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "u_cf": self.u_cf.as_dict(),
            "contact_policy": self.contact_policy,
            "language": self.language,
            "query_node_indices": (
                None
                if self.query_node_indices is None
                else list(self.query_node_indices)
            ),
        }

    @property
    def artifact_id(self) -> str:
        return _artifact_id(self.contract_type, self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
            "contract_type": self.contract_type,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CounterfactualQueryContext:
        _require_header(
            values,
            contract_type=cls.contract_type,
            payload_fields={
                "protocol_id",
                "case_id",
                "u_cf",
                "contact_policy",
                "language",
                "query_node_indices",
            },
        )
        name = f"stage-provenance {cls.contract_type}"
        policy = _require_string(values, "contact_policy", name=name)
        if policy not in {"same_grasp", "new_contact"}:
            raise ValueError("contact_policy must be 'same_grasp' or 'new_contact'")
        language = values["language"]
        if language is not None and not isinstance(language, str):
            raise ValueError(f"{name}.language must be null or a string")
        raw_nodes = values["query_node_indices"]
        if raw_nodes is None:
            nodes = None
        else:
            if not isinstance(raw_nodes, list):
                raise ValueError(
                    f"{name}.query_node_indices must be null or a JSON array"
                )
            if any(type(index) is not int for index in raw_nodes):
                raise ValueError(
                    f"{name}.query_node_indices must contain only integers"
                )
            nodes = tuple(raw_nodes)
        return cls(
            protocol_id=_require_string(values, "protocol_id", name=name),
            case_id=_require_string(values, "case_id", name=name),
            u_cf=_action_window_from_dict(
                _require_mapping(values, "u_cf", name=name),
                name=f"{name}.u_cf",
            ),
            contact_policy=cast(
                Literal["same_grasp", "new_contact"],
                policy,
            ),
            language=language,
            query_node_indices=nodes,
        )


@dataclass(frozen=True)
class EvaluationTarget:
    """Content identity for a held-out target independent of the tested query."""

    contract_type: ClassVar[str] = "EvaluationTarget"

    protocol_id: str
    target: ObservationWindow

    def __post_init__(self) -> None:
        if not self.protocol_id:
            raise ValueError("protocol_id must be nonempty")

    @property
    def case_id(self) -> str:
        return self.target.case_id

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "target": self.target.as_dict(),
        }

    @property
    def artifact_id(self) -> str:
        return _artifact_id(self.contract_type, self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE_PROVENANCE_SCHEMA_VERSION,
            "contract_type": self.contract_type,
            **self._payload(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> EvaluationTarget:
        _require_header(
            values,
            contract_type=cls.contract_type,
            payload_fields={"protocol_id", "target"},
        )
        name = f"stage-provenance {cls.contract_type}"
        return cls(
            protocol_id=_require_string(values, "protocol_id", name=name),
            target=_observation_window_from_dict(
                _require_mapping(values, "target", name=name),
                name=f"{name}.target",
            ),
        )


def build_factual_evidence_context(
    context: CausalContext,
    observations: np.ndarray,
    observed_actions: np.ndarray,
    *,
    evidence_frame_stop: int,
) -> FactualEvidenceContext:
    """Derive a factual identity without reading or hashing held-out observations."""

    if not (
        context.o_plus.frame_start < evidence_frame_stop <= context.o_plus.frame_stop
    ):
        raise ValueError("evidence_frame_stop must be a nonempty O+ prefix")
    _require_window_digest(observations, context.o_minus, name="O-")
    _require_window_digest(observed_actions, context.u_obs, name="u_obs")
    observation_array = _frame_array(
        observations,
        name="observations",
        frame_stop=evidence_frame_stop,
    )
    prefix = ObservationWindow(
        case_id=context.case_id,
        stream_id=context.o_plus.stream_id,
        frame_start=context.o_plus.frame_start,
        frame_stop=evidence_frame_stop,
        content_sha256=array_sha256(
            observation_array[context.o_plus.frame_start : evidence_frame_stop]
        ),
    )
    return FactualEvidenceContext(
        protocol_id=context.protocol_id,
        o_minus=context.o_minus,
        o_plus_prefix=prefix,
        u_obs=context.u_obs,
    )


def build_counterfactual_query_context(
    query: CounterfactualQuery,
) -> CounterfactualQueryContext:
    """Derive a query identity without retaining the V1 target-bearing context."""

    nodes = (
        None
        if query.query_node_indices is None
        else tuple(int(index) for index in query.query_node_indices)
    )
    return CounterfactualQueryContext(
        protocol_id=query.context.protocol_id,
        case_id=query.context.case_id,
        u_cf=query.context.u_cf,
        contact_policy=query.contact_policy,
        language=query.language,
        query_node_indices=nodes,
    )


def build_evaluation_target(
    context: CausalContext,
    observations: np.ndarray,
    *,
    target_frame_start: int,
) -> EvaluationTarget:
    """Derive the held-out target identity after the evaluation suffix is opened."""

    if not (
        context.o_plus.frame_start <= target_frame_start < context.o_plus.frame_stop
    ):
        raise ValueError("target_frame_start must leave a nonempty O+ suffix")
    _require_window_digest(observations, context.o_plus, name="O+")
    observation_array = _frame_array(
        observations,
        name="observations",
        frame_stop=context.o_plus.frame_stop,
    )
    target = ObservationWindow(
        case_id=context.case_id,
        stream_id=context.o_plus.stream_id,
        frame_start=target_frame_start,
        frame_stop=context.o_plus.frame_stop,
        content_sha256=array_sha256(
            observation_array[target_frame_start : context.o_plus.frame_stop]
        ),
    )
    return EvaluationTarget(protocol_id=context.protocol_id, target=target)


def validate_stage_contexts(
    factual: FactualEvidenceContext,
    query: CounterfactualQueryContext,
    target: EvaluationTarget,
) -> None:
    """Validate that independently addressed stages form one admissible chain."""

    if len({factual.protocol_id, query.protocol_id, target.protocol_id}) != 1:
        raise ValueError("stage contexts must use the same protocol")
    if len({factual.case_id, query.case_id, target.case_id}) != 1:
        raise ValueError("stage contexts must use the same case")
    if factual.o_plus_prefix.stream_id != target.target.stream_id:
        raise ValueError("held-out target must continue the factual observation stream")
    if factual.o_plus_prefix.frame_stop != target.target.frame_start:
        raise ValueError("held-out target must begin at the factual evidence boundary")
    if (
        query.u_cf.frame_start > target.target.frame_start
        or query.u_cf.frame_stop < target.target.frame_stop
    ):
        raise ValueError("u_cf must cover the held-out evaluation target")
    for name, value in (
        ("factual artifact_id", factual.artifact_id),
        ("query artifact_id", query.artifact_id),
        ("target artifact_id", target.artifact_id),
    ):
        _validate_sha256(value, name=name)
