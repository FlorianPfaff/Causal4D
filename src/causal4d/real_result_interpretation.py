"""Predeclared interpretation contract for the confirmatory real experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from causal4d.atomic_io import atomic_write_json

INTERPRETATION_SCHEMA_VERSION = 1
INTERPRETATION_CONTRACT_ID = "causal4d-real-result-interpretation-v1"
EXPECTED_PROTOCOL_ID = "causal4d-sloth-multi-action-v1"
EXPECTED_PROTOCOL_DESIGN_SHA256 = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
EXPECTED_PREACQUISITION_SHA256 = (
    "0e167538a7824e5ec053031d8359d4e9b4ff89ad61a85666400a86c2a88ac42f"
)

GateStatus = Literal["passed", "failed", "not_estimable"]
EvidenceStatus = Literal["complete", "incomplete"]
OracleDiagnosis = Literal[
    "intervention_headroom",
    "model_discrepancy_dominant",
    "mixed",
    "not_estimable",
]
PaperStatus = Literal["positive", "bounded_positive", "negative", "incomplete"]

_GATE_STATUSES = {"passed", "failed", "not_estimable"}
_EVIDENCE_STATUSES = {"complete", "incomplete"}
_ORACLE_DIAGNOSES = {
    "intervention_headroom",
    "model_discrepancy_dominant",
    "mixed",
    "not_estimable",
}

_PROHIBITED_CLAIMS = (
    "individual-level real counterfactual ground truth",
    "overall state of the art beyond the registered same-object protocol",
    "calibration of the raw physical-posterior covariance",
    "general robot-execution safety or hardware-control success",
    "Prob4D benefit without its separate prospective experiment",
    "real contact recovery without independent contact instrumentation",
)

_CONTRACT_DESCRIPTOR = {
    "schema_version": INTERPRETATION_SCHEMA_VERSION,
    "contract_id": INTERPRETATION_CONTRACT_ID,
    "protocol_id": EXPECTED_PROTOCOL_ID,
    "protocol_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
    "preacquisition_amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
    "primary_gate_order": [
        "factual_continuation",
        "same_grasp_transfer",
        "new_contact_transfer",
        "execution_block_calibration",
    ],
    "prohibited_claims": list(_PROHIBITED_CLAIMS),
}

_RULES: dict[str, dict[str, Any]] = {
    "confirmatory_boundary_violated": {
        "paper_status": "incomplete",
        "headline": "The target-informed selection boundary was violated.",
        "claims": (),
        "next_action": (
            "Report the violation and issue a new protocol before reanalysis."
        ),
    },
    "incomplete_evidence": {
        "paper_status": "incomplete",
        "headline": "The confirmatory evidence registry is incomplete.",
        "claims": (),
        "next_action": (
            "Complete or explicitly report the registered evidence without "
            "replacing missing or failed executions."
        ),
    },
    "primary_chain_not_supported": {
        "paper_status": "negative",
        "headline": "The factual-continuation gate does not pass.",
        "claims": (),
        "next_action": (
            "Report the negative result and use only preregistered oracle "
            "diagnostics to localize the failure."
        ),
    },
    "full_chain_supported": {
        "paper_status": "positive",
        "headline": "The complete registered Causal4D evidence chain passes.",
        "claims": (
            "Factual continuation and persistent/event-specific transfer pass.",
            (
                "Execution-block calibration supports the registered nominal "
                "coverage statement under its finite-sample assumptions."
            ),
        ),
        "next_action": (
            "Freeze the evidence bundle and keep the claim bounded to the "
            "registered object, actions, contacts, and calibration design."
        ),
    },
    "transfer_supported_calibration_limited": {
        "paper_status": "bounded_positive",
        "headline": "Factual and transfer gates pass, but calibration fails.",
        "claims": (
            "Factual continuation and persistent/event-specific transfer pass.",
        ),
        "next_action": (
            "Report the transfer result with the calibration limitation; do not "
            "make a calibrated-risk or hardware-safety claim."
        ),
    },
    "transfer_supported_calibration_unresolved": {
        "paper_status": "incomplete",
        "headline": "Transfer passes, but calibration is not estimable.",
        "claims": (
            "Factual continuation and persistent/event-specific transfer pass.",
        ),
        "next_action": (
            "Report bounded transfer evidence and retain the unresolved "
            "independent-execution calibration endpoint."
        ),
    },
    "persistent_transfer_only": {
        "paper_status": "bounded_positive",
        "headline": "Same-grasp transfer passes, but new-contact transfer fails.",
        "claims": (
            "Persistent realized-actuation information transfers within grasp.",
        ),
        "next_action": (
            "Report the contact/event-transfer failure and do not claim that a "
            "fresh kappa resolves new-contact interventions."
        ),
    },
    "incoherent_transfer_pattern": {
        "paper_status": "bounded_positive",
        "headline": "New-contact transfer passes without same-grasp transfer.",
        "claims": ("The registered new-contact arm shows bounded improvement.",),
        "next_action": (
            "Report the inconsistent arms separately; do not make a general "
            "realized-intervention transfer claim."
        ),
    },
    "factual_only": {
        "paper_status": "bounded_positive",
        "headline": "Factual continuation passes, but both transfer gates fail.",
        "claims": (
            "Prefix-conditioned factual continuation improves under the protocol.",
        ),
        "next_action": (
            "Do not describe the realized-intervention posterior as transferable "
            "across actions or contacts."
        ),
    },
    "partial_transfer_evidence": {
        "paper_status": "incomplete",
        "headline": "At least one transfer endpoint is not estimable.",
        "claims": ("Factual continuation passes under the protocol.",),
        "next_action": (
            "Report every non-estimable endpoint without inferring the complete "
            "transfer chain from the available subset."
        ),
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: str, *, name: str) -> str:
    _require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{name} must be a nonnegative integer",
    )
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def interpretation_contract_sha256() -> str:
    return _canonical_sha256(_CONTRACT_DESCRIPTOR)


@dataclass(frozen=True)
class RealResultGateSummary:
    """Registered gate outcomes and provenance for one confirmatory analysis."""

    protocol_id: str
    protocol_design_sha256: str
    preacquisition_amendment_sha256: str
    method_freeze_sha256: str
    analysis_manifest_sha256: str
    evidence_status: EvidenceStatus
    factual_continuation: GateStatus
    same_grasp_transfer: GateStatus
    new_contact_transfer: GateStatus
    execution_block_calibration: GateStatus
    oracle_diagnosis: OracleDiagnosis = "not_estimable"
    technical_failure_count: int = 0
    preregistered_exclusion_count: int = 0
    target_informed_selection: bool = False

    def __post_init__(self) -> None:
        _require(
            self.protocol_id == EXPECTED_PROTOCOL_ID,
            "gate summary targets an unexpected protocol",
        )
        _sha256(self.protocol_design_sha256, name="protocol_design_sha256")
        _require(
            self.protocol_design_sha256 == EXPECTED_PROTOCOL_DESIGN_SHA256,
            "gate summary protocol digest does not match the locked design",
        )
        _sha256(
            self.preacquisition_amendment_sha256,
            name="preacquisition_amendment_sha256",
        )
        _require(
            self.preacquisition_amendment_sha256
            == EXPECTED_PREACQUISITION_SHA256,
            "gate summary amendment digest does not match locked v4 amendment",
        )
        _sha256(self.method_freeze_sha256, name="method_freeze_sha256")
        _sha256(self.analysis_manifest_sha256, name="analysis_manifest_sha256")
        _require(self.evidence_status in _EVIDENCE_STATUSES, "unknown evidence status")
        for name, value in (
            ("factual_continuation", self.factual_continuation),
            ("same_grasp_transfer", self.same_grasp_transfer),
            ("new_contact_transfer", self.new_contact_transfer),
            ("execution_block_calibration", self.execution_block_calibration),
        ):
            _require(value in _GATE_STATUSES, f"unknown {name} status")
        _require(
            self.oracle_diagnosis in _ORACLE_DIAGNOSES,
            "unknown oracle diagnosis",
        )
        _nonnegative_integer(
            self.technical_failure_count,
            name="technical_failure_count",
        )
        _nonnegative_integer(
            self.preregistered_exclusion_count,
            name="preregistered_exclusion_count",
        )
        _require(
            isinstance(self.target_informed_selection, bool),
            "target_informed_selection must be Boolean",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "protocol_design_sha256": self.protocol_design_sha256,
            "preacquisition_amendment_sha256": (
                self.preacquisition_amendment_sha256
            ),
            "method_freeze_sha256": self.method_freeze_sha256,
            "analysis_manifest_sha256": self.analysis_manifest_sha256,
            "evidence_status": self.evidence_status,
            "factual_continuation": self.factual_continuation,
            "same_grasp_transfer": self.same_grasp_transfer,
            "new_contact_transfer": self.new_contact_transfer,
            "execution_block_calibration": self.execution_block_calibration,
            "oracle_diagnosis": self.oracle_diagnosis,
            "technical_failure_count": self.technical_failure_count,
            "preregistered_exclusion_count": self.preregistered_exclusion_count,
            "target_informed_selection": self.target_informed_selection,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> RealResultGateSummary:
        _require(values.get("schema_version") == 1, "unsupported gate-summary schema")
        _require(
            values.get("artifact_kind") == "Causal4DRealResultGateSummary",
            "unexpected gate-summary artifact kind",
        )
        return cls(
            protocol_id=str(values["protocol_id"]),
            protocol_design_sha256=str(values["protocol_design_sha256"]),
            preacquisition_amendment_sha256=str(
                values["preacquisition_amendment_sha256"]
            ),
            method_freeze_sha256=str(values["method_freeze_sha256"]),
            analysis_manifest_sha256=str(values["analysis_manifest_sha256"]),
            evidence_status=cast(EvidenceStatus, values["evidence_status"]),
            factual_continuation=cast(GateStatus, values["factual_continuation"]),
            same_grasp_transfer=cast(GateStatus, values["same_grasp_transfer"]),
            new_contact_transfer=cast(GateStatus, values["new_contact_transfer"]),
            execution_block_calibration=cast(
                GateStatus, values["execution_block_calibration"]
            ),
            oracle_diagnosis=cast(
                OracleDiagnosis, values.get("oracle_diagnosis", "not_estimable")
            ),
            technical_failure_count=_nonnegative_integer(
                values.get("technical_failure_count", 0),
                name="technical_failure_count",
            ),
            preregistered_exclusion_count=_nonnegative_integer(
                values.get("preregistered_exclusion_count", 0),
                name="preregistered_exclusion_count",
            ),
            target_informed_selection=values.get("target_informed_selection", False),
        )


@dataclass(frozen=True)
class RealResultInterpretation:
    gates: RealResultGateSummary
    rule_id: str
    paper_status: PaperStatus
    headline: str
    supported_claims: tuple[str, ...]
    required_limitations: tuple[str, ...]
    prohibited_claims: tuple[str, ...]
    next_action: str

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": INTERPRETATION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DRealResultInterpretation",
            "interpretation_contract_id": INTERPRETATION_CONTRACT_ID,
            "interpretation_contract_sha256": interpretation_contract_sha256(),
            "gates": self.gates.as_dict(),
            "rule_id": self.rule_id,
            "classification": self.rule_id,
            "paper_status": self.paper_status,
            "headline": self.headline,
            "supported_claims": list(self.supported_claims),
            "required_limitations": list(self.required_limitations),
            "prohibited_claims": list(self.prohibited_claims),
            "next_action": self.next_action,
        }

    @property
    def result_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["result_sha256"] = self.result_sha256
        return payload


def _select_rule(gates: RealResultGateSummary) -> str:
    if gates.target_informed_selection:
        return "confirmatory_boundary_violated"
    if gates.evidence_status == "incomplete":
        return "incomplete_evidence"
    if gates.factual_continuation != "passed":
        return "primary_chain_not_supported"
    same_grasp = gates.same_grasp_transfer
    new_contact = gates.new_contact_transfer
    calibration = gates.execution_block_calibration
    if same_grasp == "passed" and new_contact == "passed":
        if calibration == "passed":
            return "full_chain_supported"
        if calibration == "failed":
            return "transfer_supported_calibration_limited"
        return "transfer_supported_calibration_unresolved"
    if same_grasp == "passed" and new_contact == "failed":
        return "persistent_transfer_only"
    if same_grasp == "failed" and new_contact == "passed":
        return "incoherent_transfer_pattern"
    if same_grasp == "failed" and new_contact == "failed":
        return "factual_only"
    return "partial_transfer_evidence"


def _limitations(gates: RealResultGateSummary, rule_id: str) -> tuple[str, ...]:
    values = [
        (
            "The real result is held-out interventional prediction from matched "
            "initial conditions, not individual counterfactual ground truth."
        ),
        "The claim is bounded to the registered same-object protocol.",
        (
            "Semantic, planning, and public-data branches cannot rescue a failed "
            "primary gate."
        ),
    ]
    if gates.technical_failure_count:
        values.append(
            f"The report retains {gates.technical_failure_count} technical failures."
        )
    if gates.preregistered_exclusion_count:
        values.append(
            "The report retains "
            f"{gates.preregistered_exclusion_count} preregistered exclusions."
        )
    if gates.oracle_diagnosis == "model_discrepancy_dominant":
        values.append(
            "The oracle diagnosis attributes dominant remaining headroom to "
            "physical/model discrepancy rather than proposal width."
        )
    elif gates.oracle_diagnosis == "intervention_headroom":
        values.append(
            "The oracle diagnosis leaves material intervention-inference or "
            "proposal-support headroom."
        )
    elif gates.oracle_diagnosis == "mixed":
        values.append("The oracle diagnosis does not isolate one failure source.")
    if (
        gates.execution_block_calibration == "passed"
        and rule_id != "full_chain_supported"
    ):
        values.append(
            "Passing calibration does not rescue a failed factual or transfer gate."
        )
    return tuple(values)


def interpret_real_result(gates: RealResultGateSummary) -> RealResultInterpretation:
    """Apply the locked tree without reading raw target outcomes."""

    rule_id = _select_rule(gates)
    rule = _RULES[rule_id]
    return RealResultInterpretation(
        gates=gates,
        rule_id=rule_id,
        paper_status=cast(PaperStatus, rule["paper_status"]),
        headline=str(rule["headline"]),
        supported_claims=tuple(rule["claims"]),
        required_limitations=_limitations(gates, rule_id),
        prohibited_claims=_PROHIBITED_CLAIMS,
        next_action=str(rule["next_action"]),
    )


def validate_real_result_interpretation(
    payload: Mapping[str, Any],
) -> RealResultInterpretation:
    """Recompute the artifact and reject any edited interpretation."""

    _require(payload.get("schema_version") == 1, "unsupported interpretation schema")
    _require(
        payload.get("artifact_kind") == "Causal4DRealResultInterpretation",
        "unexpected interpretation artifact kind",
    )
    gates_payload = payload.get("gates")
    _require(isinstance(gates_payload, Mapping), "interpretation gates are missing")
    gates = RealResultGateSummary(
        protocol_id=str(gates_payload["protocol_id"]),
        protocol_design_sha256=str(gates_payload["protocol_design_sha256"]),
        preacquisition_amendment_sha256=str(
            gates_payload["preacquisition_amendment_sha256"]
        ),
        method_freeze_sha256=str(gates_payload["method_freeze_sha256"]),
        analysis_manifest_sha256=str(gates_payload["analysis_manifest_sha256"]),
        evidence_status=cast(EvidenceStatus, gates_payload["evidence_status"]),
        factual_continuation=cast(
            GateStatus, gates_payload["factual_continuation"]
        ),
        same_grasp_transfer=cast(
            GateStatus, gates_payload["same_grasp_transfer"]
        ),
        new_contact_transfer=cast(
            GateStatus, gates_payload["new_contact_transfer"]
        ),
        execution_block_calibration=cast(
            GateStatus, gates_payload["execution_block_calibration"]
        ),
        oracle_diagnosis=cast(OracleDiagnosis, gates_payload["oracle_diagnosis"]),
        technical_failure_count=_nonnegative_integer(
            gates_payload["technical_failure_count"],
            name="technical_failure_count",
        ),
        preregistered_exclusion_count=_nonnegative_integer(
            gates_payload["preregistered_exclusion_count"],
            name="preregistered_exclusion_count",
        ),
        target_informed_selection=gates_payload["target_informed_selection"],
    )
    expected = interpret_real_result(gates)
    _require(
        dict(payload) == expected.as_dict(),
        "serialized interpretation differs from the locked decision tree",
    )
    return expected


def load_real_result_interpretation(
    path: str | Path,
) -> RealResultInterpretation:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "interpretation JSON must be an object")
    return validate_real_result_interpretation(payload)


def write_real_result_interpretation(
    path: str | Path,
    interpretation: RealResultInterpretation,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    atomic_write_json(output, interpretation.as_dict(), overwrite=overwrite)
    return output


__all__ = [
    "EXPECTED_PREACQUISITION_SHA256",
    "EXPECTED_PROTOCOL_DESIGN_SHA256",
    "EXPECTED_PROTOCOL_ID",
    "INTERPRETATION_CONTRACT_ID",
    "INTERPRETATION_SCHEMA_VERSION",
    "RealResultGateSummary",
    "RealResultInterpretation",
    "interpret_real_result",
    "interpretation_contract_sha256",
    "load_real_result_interpretation",
    "validate_real_result_interpretation",
    "write_real_result_interpretation",
]
