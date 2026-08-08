"""Fail-closed ownership accounting for evidence consumed across inference stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal, cast

import numpy as np

from causal4d.contracts import FactualIntervention
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.sensor_evidence import ActuatorEvidence, ContactWrenchEvidence
from causal4d.sensor_factorized_abduction import (
    IndependentSensorAbductionConfig,
    reweight_factual_intervention_with_independent_sensors,
)


EVIDENCE_OWNERSHIP_SCHEMA_VERSION = 1
EVIDENCE_OWNERSHIP_ARTIFACT_KIND = "ConsumedEvidenceLedger"
EvidenceRole = Literal[
    "state_update",
    "actuator_abduction",
    "contact_abduction",
    "joint_state_intervention_update",
    "calibration_only",
    "evaluation_only",
]
_ROLES = frozenset(
    {
        "state_update",
        "actuator_abduction",
        "contact_abduction",
        "joint_state_intervention_update",
        "calibration_only",
        "evaluation_only",
    }
)
_INDEPENDENT_POSTERIOR_ROLES = frozenset(
    {"state_update", "actuator_abduction", "contact_abduction"}
)
_JOINT_ROLE = "joint_state_intervention_update"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    result = _require_nonempty_string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _require_frame(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_exact_fields(
    values: Any,
    *,
    required: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping) or any(type(key) is not str for key in values):
        raise ValueError(f"{name} must be a string-keyed mapping")
    actual = set(values)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return cast(Mapping[str, Any], values)


@dataclass(frozen=True)
class EvidenceConsumptionV1:
    """One content-addressed use of a raw measurement factor."""

    evidence_id: str
    raw_factor_id: str
    source_repository: str
    source_revision: str
    sensor_family: str
    stream_id: str
    clock_id: str
    correlation_group_id: str
    frame_start: int
    frame_stop: int
    role: EvidenceRole
    source_file_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("evidence_id", "raw_factor_id"):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        for name in (
            "source_repository",
            "source_revision",
            "sensor_family",
            "stream_id",
            "clock_id",
            "correlation_group_id",
        ):
            object.__setattr__(
                self,
                name,
                _require_nonempty_string(getattr(self, name), name=name),
            )
        frame_start = _require_frame(self.frame_start, name="frame_start")
        frame_stop = _require_frame(self.frame_stop, name="frame_stop", minimum=1)
        if frame_stop <= frame_start:
            raise ValueError("evidence interval must be nonempty")
        object.__setattr__(self, "frame_start", frame_start)
        object.__setattr__(self, "frame_stop", frame_stop)
        if type(self.role) is not str or self.role not in _ROLES:
            raise ValueError(f"unsupported evidence-consumption role: {self.role!r}")
        if self.source_file_sha256 is not None:
            object.__setattr__(
                self,
                "source_file_sha256",
                _require_sha256(
                    self.source_file_sha256,
                    name="source_file_sha256",
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="evidence-consumption metadata must be finite JSON",
            ),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "raw_factor_id": self.raw_factor_id,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "sensor_family": self.sensor_family,
            "stream_id": self.stream_id,
            "clock_id": self.clock_id,
            "correlation_group_id": self.correlation_group_id,
            "frame_start": self.frame_start,
            "frame_stop": self.frame_stop,
            "role": self.role,
            "source_file_sha256": self.source_file_sha256,
            "metadata": plain_json(self.metadata),
        }

    @property
    def consumption_id(self) -> str:
        return _sha256_payload(
            {
                "schema_version": EVIDENCE_OWNERSHIP_SCHEMA_VERSION,
                "artifact_kind": "EvidenceConsumption",
                **self._identity_payload(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "consumption_id": self.consumption_id,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> EvidenceConsumptionV1:
        fields = _require_exact_fields(
            values,
            name="evidence consumption",
            required=frozenset(
                {
                    "evidence_id",
                    "raw_factor_id",
                    "source_repository",
                    "source_revision",
                    "sensor_family",
                    "stream_id",
                    "clock_id",
                    "correlation_group_id",
                    "frame_start",
                    "frame_stop",
                    "role",
                    "source_file_sha256",
                    "metadata",
                    "consumption_id",
                }
            ),
        )
        record = cls(
            evidence_id=fields["evidence_id"],
            raw_factor_id=fields["raw_factor_id"],
            source_repository=fields["source_repository"],
            source_revision=fields["source_revision"],
            sensor_family=fields["sensor_family"],
            stream_id=fields["stream_id"],
            clock_id=fields["clock_id"],
            correlation_group_id=fields["correlation_group_id"],
            frame_start=fields["frame_start"],
            frame_stop=fields["frame_stop"],
            role=fields["role"],
            source_file_sha256=fields["source_file_sha256"],
            metadata=fields["metadata"],
        )
        if fields["consumption_id"] != record.consumption_id:
            raise ValueError("evidence-consumption content identity changed")
        return record


@dataclass(frozen=True)
class ConsumedEvidenceLedgerV1:
    """Immutable accounting of all evidence admitted into one causal posterior."""

    protocol_id: str
    case_id: str
    causal_frame_stop: int
    entries: tuple[EvidenceConsumptionV1, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_id",
            _require_nonempty_string(self.protocol_id, name="protocol_id"),
        )
        object.__setattr__(
            self,
            "case_id",
            _require_nonempty_string(self.case_id, name="case_id"),
        )
        causal_frame_stop = _require_frame(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
        if isinstance(self.entries, (str, bytes)) or not isinstance(
            self.entries, Sequence
        ):
            raise ValueError("entries must be a sequence of EvidenceConsumptionV1")
        normalized = tuple(self.entries)
        if any(type(entry) is not EvidenceConsumptionV1 for entry in normalized):
            raise ValueError("entries must contain only EvidenceConsumptionV1")
        normalized = tuple(
            sorted(
                normalized,
                key=lambda entry: (
                    entry.evidence_id,
                    entry.role,
                    entry.stream_id,
                    entry.consumption_id,
                ),
            )
        )
        for entry in normalized:
            if entry.frame_stop > causal_frame_stop:
                raise ValueError("evidence consumption crosses the causal frame stop")
        self._validate_unique_ownership(normalized)
        self._validate_correlation_groups(normalized)
        object.__setattr__(self, "entries", normalized)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="evidence-ledger metadata must be finite JSON",
            ),
        )

    @staticmethod
    def _validate_unique_ownership(entries: tuple[EvidenceConsumptionV1, ...]) -> None:
        evidence_ids: set[str] = set()
        raw_factor_ids: set[str] = set()
        source_files: dict[str, str] = {}
        for entry in entries:
            if entry.evidence_id in evidence_ids:
                raise ValueError("one evidence artifact is consumed more than once")
            evidence_ids.add(entry.evidence_id)
            if entry.raw_factor_id in raw_factor_ids:
                raise ValueError(
                    "one raw measurement factor is consumed more than once"
                )
            raw_factor_ids.add(entry.raw_factor_id)
            if entry.source_file_sha256 is None:
                continue
            previous = source_files.setdefault(
                entry.source_file_sha256,
                entry.evidence_id,
            )
            if previous != entry.evidence_id:
                raise ValueError(
                    "identical source bytes were relabelled as distinct evidence"
                )

    @staticmethod
    def _validate_correlation_groups(
        entries: tuple[EvidenceConsumptionV1, ...],
    ) -> None:
        groups: dict[str, set[str]] = {}
        for entry in entries:
            if entry.role in _ROLES - {"calibration_only", "evaluation_only"}:
                groups.setdefault(entry.correlation_group_id, set()).add(entry.role)
        for group_id, roles in groups.items():
            if _JOINT_ROLE in roles and len(roles) > 1:
                raise ValueError(
                    "joint evidence cannot also be consumed independently in "
                    f"correlation group {group_id!r}"
                )
            independent_roles = roles & _INDEPENDENT_POSTERIOR_ROLES
            if len(independent_roles) > 1:
                rendered = ", ".join(sorted(independent_roles))
                raise ValueError(
                    "correlated evidence was multiplied across inference stages in "
                    f"group {group_id!r}: {rendered}"
                )

    @property
    def artifact_id(self) -> str:
        return _sha256_payload(
            {
                "schema_version": EVIDENCE_OWNERSHIP_SCHEMA_VERSION,
                "artifact_kind": EVIDENCE_OWNERSHIP_ARTIFACT_KIND,
                "protocol_id": self.protocol_id,
                "case_id": self.case_id,
                "causal_frame_stop": self.causal_frame_stop,
                "entries": [entry.as_dict() for entry in self.entries],
                "metadata": plain_json(self.metadata),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_OWNERSHIP_SCHEMA_VERSION,
            "artifact_kind": EVIDENCE_OWNERSHIP_ARTIFACT_KIND,
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "causal_frame_stop": self.causal_frame_stop,
            "entries": [entry.as_dict() for entry in self.entries],
            "metadata": plain_json(self.metadata),
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ConsumedEvidenceLedgerV1:
        fields = _require_exact_fields(
            values,
            name="consumed evidence ledger",
            required=frozenset(
                {
                    "schema_version",
                    "artifact_kind",
                    "protocol_id",
                    "case_id",
                    "causal_frame_stop",
                    "entries",
                    "metadata",
                    "artifact_id",
                }
            ),
        )
        if fields["schema_version"] != EVIDENCE_OWNERSHIP_SCHEMA_VERSION:
            raise ValueError("unsupported evidence-ownership schema version")
        if fields["artifact_kind"] != EVIDENCE_OWNERSHIP_ARTIFACT_KIND:
            raise ValueError("unsupported evidence-ownership artifact kind")
        raw_entries = fields["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError("evidence-ledger entries must be a JSON array")
        ledger = cls(
            protocol_id=fields["protocol_id"],
            case_id=fields["case_id"],
            causal_frame_stop=fields["causal_frame_stop"],
            entries=tuple(
                EvidenceConsumptionV1.from_dict(entry) for entry in raw_entries
            ),
            metadata=fields["metadata"],
        )
        if fields["artifact_id"] != ledger.artifact_id:
            raise ValueError("evidence-ledger content identity changed")
        return ledger

    def extend(
        self,
        *entries: EvidenceConsumptionV1,
    ) -> ConsumedEvidenceLedgerV1:
        """Return a new ledger after validating the combined ownership boundary."""

        return ConsumedEvidenceLedgerV1(
            protocol_id=self.protocol_id,
            case_id=self.case_id,
            causal_frame_stop=self.causal_frame_stop,
            entries=self.entries + tuple(entries),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class OwnedSensorAbductionResult:
    """Strict sensor update together with its resulting evidence ledger."""

    factual_intervention: FactualIntervention
    evidence_ledger: ConsumedEvidenceLedgerV1
    sensor_update_applied: bool


def evidence_consumption_for_independent_sensor(
    evidence: ActuatorEvidence | ContactWrenchEvidence,
    *,
    source_repository: str,
    source_revision: str,
    correlation_group_id: str,
    raw_factor_id: str | None = None,
    source_file_sha256: str | None = None,
) -> EvidenceConsumptionV1:
    """Describe one independent-sensor factor for the strict ownership path."""

    if type(evidence) is ActuatorEvidence:
        sensor_family = "actuator_state"
        role: EvidenceRole = "actuator_abduction"
    elif type(evidence) is ContactWrenchEvidence:
        sensor_family = "contact_wrench"
        role = "contact_abduction"
    else:
        raise TypeError("unsupported independent-sensor evidence type")
    return EvidenceConsumptionV1(
        evidence_id=evidence.artifact_id,
        raw_factor_id=raw_factor_id or evidence.artifact_id,
        source_repository=source_repository,
        source_revision=source_revision,
        sensor_family=sensor_family,
        stream_id=evidence.stream_id,
        clock_id=evidence.clock_id,
        correlation_group_id=correlation_group_id,
        frame_start=0,
        frame_stop=evidence.evidence_frame_stop,
        role=role,
        source_file_sha256=source_file_sha256,
        metadata={"provenance": evidence.provenance},
    )


def _validate_ledger_binding(
    factual: FactualIntervention,
    ledger: ConsumedEvidenceLedgerV1,
) -> None:
    if ledger.protocol_id != factual.context.protocol_id:
        raise ValueError("evidence ledger identifies a different protocol")
    if ledger.case_id != factual.context.case_id:
        raise ValueError("evidence ledger identifies a different case")
    if ledger.causal_frame_stop != factual.evidence_frame_stop:
        raise ValueError("evidence ledger and factual prefix stops differ")

    embedded = factual.metadata.get("consumed_evidence_ledger")
    if embedded is None:
        return
    if not isinstance(embedded, Mapping):
        raise ValueError("factual intervention embeds an invalid evidence ledger")
    try:
        embedded_payload = plain_json(embedded)
        if not isinstance(embedded_payload, dict):
            raise ValueError("embedded ledger is not a JSON object")
        embedded_ledger = ConsumedEvidenceLedgerV1.from_dict(embedded_payload)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "factual intervention embeds an invalid evidence ledger"
        ) from error
    if embedded_ledger.as_dict() != ledger.as_dict():
        raise ValueError(
            "supplied prior evidence ledger differs from the ledger embedded "
            "in the factual intervention"
        )


def _validate_consumption_binding(
    consumption: EvidenceConsumptionV1 | None,
    evidence: ActuatorEvidence | ContactWrenchEvidence | None,
    *,
    expected_role: EvidenceRole,
    expected_sensor_family: str,
    name: str,
) -> None:
    if evidence is None:
        if consumption is not None:
            raise ValueError(f"{name} consumption requires {name} evidence")
        return
    if consumption is None:
        raise ValueError(f"{name} evidence requires an ownership consumption record")
    expected = (
        evidence.artifact_id,
        evidence.stream_id,
        evidence.clock_id,
        evidence.evidence_frame_stop,
        expected_role,
        expected_sensor_family,
    )
    supplied = (
        consumption.evidence_id,
        consumption.stream_id,
        consumption.clock_id,
        consumption.frame_stop,
        consumption.role,
        consumption.sensor_family,
    )
    if supplied != expected or consumption.frame_start != 0:
        raise ValueError(f"{name} evidence and ownership record differ")


def strict_reweight_factual_intervention_with_independent_sensors(
    factual: FactualIntervention,
    *,
    prior_evidence_ledger: ConsumedEvidenceLedgerV1,
    actuator_evidence: ActuatorEvidence | None = None,
    actuator_consumption: EvidenceConsumptionV1 | None = None,
    predicted_actuator_positions_m: np.ndarray | None = None,
    predicted_actuator_variance_m2: np.ndarray | None = None,
    wrench_evidence: ContactWrenchEvidence | None = None,
    wrench_consumption: EvidenceConsumptionV1 | None = None,
    predicted_contact_wrench: np.ndarray | None = None,
    predicted_wrench_variance: np.ndarray | None = None,
    config: IndependentSensorAbductionConfig | None = None,
) -> OwnedSensorAbductionResult:
    """Apply independent sensor factors only after ownership validation.

    This opt-in path leaves the frozen estimator untouched. It first proves that
    no raw factor, source bytes, or correlation group has already influenced an
    incompatible inference stage. Only informative factors are appended to the
    returned ledger.
    """

    _validate_ledger_binding(factual, prior_evidence_ledger)
    _validate_consumption_binding(
        actuator_consumption,
        actuator_evidence,
        expected_role="actuator_abduction",
        expected_sensor_family="actuator_state",
        name="actuator",
    )
    _validate_consumption_binding(
        wrench_consumption,
        wrench_evidence,
        expected_role="contact_abduction",
        expected_sensor_family="contact_wrench",
        name="wrench",
    )
    supplied_consumptions = tuple(
        consumption
        for consumption in (actuator_consumption, wrench_consumption)
        if consumption is not None
    )
    # Validate the complete proposed update before any likelihood is evaluated.
    prior_evidence_ledger.extend(*supplied_consumptions)
    updated = reweight_factual_intervention_with_independent_sensors(
        factual,
        actuator_evidence=actuator_evidence,
        predicted_actuator_positions_m=predicted_actuator_positions_m,
        predicted_actuator_variance_m2=predicted_actuator_variance_m2,
        wrench_evidence=wrench_evidence,
        predicted_contact_wrench=predicted_contact_wrench,
        predicted_wrench_variance=predicted_wrench_variance,
        config=config,
    )
    if updated is factual:
        return OwnedSensorAbductionResult(
            factual_intervention=factual,
            evidence_ledger=prior_evidence_ledger,
            sensor_update_applied=False,
        )
    diagnostics = updated.metadata.get("independent_sensor_abduction")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("sensor update omitted its evidence diagnostics")
    summaries = diagnostics.get("factors")
    if isinstance(summaries, (str, bytes)) or not isinstance(summaries, Sequence):
        raise RuntimeError("sensor update omitted its factor diagnostics")
    informative_ids = {
        summary.get("evidence_id")
        for summary in summaries
        if isinstance(summary, Mapping) and summary.get("informative") is True
    }
    informative_consumptions = tuple(
        consumption
        for consumption in supplied_consumptions
        if consumption.evidence_id in informative_ids
    )
    if len(informative_consumptions) != len(informative_ids):
        raise RuntimeError("sensor update reported an unbound evidence factor")
    next_ledger = prior_evidence_ledger.extend(*informative_consumptions)
    metadata = dict(updated.metadata)
    metadata["consumed_evidence_ledger"] = next_ledger.as_dict()
    bound = FactualIntervention(
        context=updated.context,
        component_ids=updated.component_ids,
        phi_names=updated.phi_names,
        kappa_names=updated.kappa_names,
        phi=updated.phi,
        kappa_obs=updated.kappa_obs,
        hypothesis_indices=updated.hypothesis_indices,
        twin_particle_indices=updated.twin_particle_indices,
        weights=updated.weights,
        evidence_frame_stop=updated.evidence_frame_stop,
        source_twin_belief_id=updated.source_twin_belief_id,
        metadata=metadata,
    )
    return OwnedSensorAbductionResult(
        factual_intervention=bound,
        evidence_ledger=next_ledger,
        sensor_update_applied=True,
    )


__all__ = [
    "ConsumedEvidenceLedgerV1",
    "EVIDENCE_OWNERSHIP_ARTIFACT_KIND",
    "EVIDENCE_OWNERSHIP_SCHEMA_VERSION",
    "EvidenceConsumptionV1",
    "EvidenceRole",
    "OwnedSensorAbductionResult",
    "evidence_consumption_for_independent_sensor",
    "strict_reweight_factual_intervention_with_independent_sensors",
]
