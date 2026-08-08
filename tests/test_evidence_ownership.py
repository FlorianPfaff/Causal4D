from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from causal4d.contracts import FactualIntervention, array_sha256, build_causal_context
from causal4d.evidence_ownership import (
    ConsumedEvidenceLedgerV1,
    EvidenceConsumptionV1,
    EvidenceRole,
    evidence_consumption_for_independent_sensor,
    strict_reweight_factual_intervention_with_independent_sensors,
)
from causal4d.sensor_evidence import ActuatorEvidence


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _consumption(
    label: str,
    *,
    role: EvidenceRole = "state_update",
    group: str = "group-a",
    source_file_sha256: str | None = None,
) -> EvidenceConsumptionV1:
    return EvidenceConsumptionV1(
        evidence_id=_digest(f"evidence:{label}"),
        raw_factor_id=_digest(f"raw:{label}"),
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        sensor_family="object_observation",
        stream_id=f"stream-{label}",
        clock_id="shared-clock",
        correlation_group_id=group,
        frame_start=0,
        frame_stop=6,
        role=role,
        source_file_sha256=source_file_sha256,
        metadata={"label": label},
    )


def _factual() -> FactualIntervention:
    observations = np.zeros((8, 2, 3), dtype=float)
    actions = np.zeros((8, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="evidence_ownership_unit",
        case_id="unit_case",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=4,
    )
    return FactualIntervention(
        context=context,
        component_ids=("z0", "z1"),
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        kappa_names=("contact_node", "slip_fraction"),
        phi=np.asarray([[1.0, 0.0, 0.0], [0.8, 1.0, 0.0]]),
        kappa_obs=np.asarray([[0.0, 0.0], [1.0, 0.2]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        weights=np.asarray([0.5, 0.5]),
        evidence_frame_stop=6,
        source_twin_belief_id=array_sha256(np.zeros(1)),
    )


def _actuator_evidence() -> ActuatorEvidence:
    positions = np.zeros((2, 1, 3), dtype=float)
    return ActuatorEvidence(
        protocol_id="evidence_ownership_unit",
        case_id="unit_case",
        observed_action_id="u_obs",
        stream_id="measured_end_effector",
        clock_id="shared-clock",
        provenance="robot encoder independent of object reconstruction",
        sample_times_s=np.asarray([0.0, 1.0 / 30.0]),
        positions_m=positions,
        variance_m2=np.full_like(positions, 1.0e-4),
        evidence_frame_stop=6,
    )


def _empty_ledger() -> ConsumedEvidenceLedgerV1:
    return ConsumedEvidenceLedgerV1(
        protocol_id="evidence_ownership_unit",
        case_id="unit_case",
        causal_frame_stop=6,
    )


def test_ledger_is_order_invariant_and_round_trips_exactly() -> None:
    first = _consumption("first")
    second = _consumption("second")
    forward = ConsumedEvidenceLedgerV1(
        protocol_id="protocol",
        case_id="case",
        causal_frame_stop=6,
        entries=(first, second),
    )
    reverse = ConsumedEvidenceLedgerV1(
        protocol_id="protocol",
        case_id="case",
        causal_frame_stop=6,
        entries=(second, first),
    )

    assert forward.artifact_id == reverse.artifact_id
    restored = ConsumedEvidenceLedgerV1.from_dict(forward.as_dict())
    assert restored.artifact_id == forward.artifact_id
    assert restored.as_dict() == forward.as_dict()


def test_ledger_rejects_duplicate_raw_factor_and_relabelled_source_bytes() -> None:
    first = _consumption("first", source_file_sha256=_digest("same-file"))
    duplicate_raw = replace(
        _consumption("second"),
        raw_factor_id=first.raw_factor_id,
    )
    with pytest.raises(ValueError, match="raw measurement factor"):
        ConsumedEvidenceLedgerV1(
            protocol_id="protocol",
            case_id="case",
            causal_frame_stop=6,
            entries=(first, duplicate_raw),
        )

    relabelled = _consumption(
        "relabelled",
        source_file_sha256=first.source_file_sha256,
    )
    with pytest.raises(ValueError, match="source bytes were relabelled"):
        ConsumedEvidenceLedgerV1(
            protocol_id="protocol",
            case_id="case",
            causal_frame_stop=6,
            entries=(first, relabelled),
        )


def test_ledger_rejects_correlated_cross_stage_multiplication() -> None:
    state = _consumption("state", role="state_update", group="shared")
    contact = _consumption(
        "contact",
        role="contact_abduction",
        group="shared",
    )

    with pytest.raises(ValueError, match="across inference stages"):
        ConsumedEvidenceLedgerV1(
            protocol_id="protocol",
            case_id="case",
            causal_frame_stop=6,
            entries=(state, contact),
        )


def test_strict_sensor_update_binds_only_informative_evidence() -> None:
    factual = _factual()
    evidence = _actuator_evidence()
    consumption = evidence_consumption_for_independent_sensor(
        evidence,
        source_repository="robot/acquisition",
        source_revision="session-v1",
        correlation_group_id="independent-actuator",
    )
    matching = evidence.positions_m
    predictions = np.stack((matching, matching + 0.2), axis=0)

    result = strict_reweight_factual_intervention_with_independent_sensors(
        factual,
        prior_evidence_ledger=_empty_ledger(),
        actuator_evidence=evidence,
        actuator_consumption=consumption,
        predicted_actuator_positions_m=predictions,
    )

    assert result.sensor_update_applied
    assert result.factual_intervention.weights[0] > 0.999
    assert result.evidence_ledger.entries == (consumption,)
    embedded = result.factual_intervention.metadata["consumed_evidence_ledger"]
    assert embedded["artifact_id"] == result.evidence_ledger.artifact_id


def test_strict_sensor_update_rejects_prior_correlated_state_use() -> None:
    factual = _factual()
    evidence = _actuator_evidence()
    state = _consumption("state", role="state_update", group="shared")
    prior = ConsumedEvidenceLedgerV1(
        protocol_id="evidence_ownership_unit",
        case_id="unit_case",
        causal_frame_stop=6,
        entries=(state,),
    )
    consumption = evidence_consumption_for_independent_sensor(
        evidence,
        source_repository="robot/acquisition",
        source_revision="session-v1",
        correlation_group_id="shared",
    )
    predictions = np.stack(
        (evidence.positions_m, evidence.positions_m + 0.2),
        axis=0,
    )

    with pytest.raises(ValueError, match="across inference stages"):
        strict_reweight_factual_intervention_with_independent_sensors(
            factual,
            prior_evidence_ledger=prior,
            actuator_evidence=evidence,
            actuator_consumption=consumption,
            predicted_actuator_positions_m=predictions,
        )


def test_uninformative_strict_sensor_factor_preserves_artifact_and_ledger() -> None:
    factual = _factual()
    evidence = _actuator_evidence()
    prior = _empty_ledger()
    consumption = evidence_consumption_for_independent_sensor(
        evidence,
        source_repository="robot/acquisition",
        source_revision="session-v1",
        correlation_group_id="independent-actuator",
    )
    predictions = np.stack((evidence.positions_m, evidence.positions_m), axis=0)

    result = strict_reweight_factual_intervention_with_independent_sensors(
        factual,
        prior_evidence_ledger=prior,
        actuator_evidence=evidence,
        actuator_consumption=consumption,
        predicted_actuator_positions_m=predictions,
    )

    assert not result.sensor_update_applied
    assert result.factual_intervention is factual
    assert result.evidence_ledger is prior


def test_strict_sensor_update_rejects_ledger_rollback() -> None:
    factual = _factual()
    evidence = _actuator_evidence()
    consumption = evidence_consumption_for_independent_sensor(
        evidence,
        source_repository="robot/acquisition",
        source_revision="session-v1",
        correlation_group_id="independent-actuator",
    )
    predictions = np.stack(
        (evidence.positions_m, evidence.positions_m + 0.2),
        axis=0,
    )
    first = strict_reweight_factual_intervention_with_independent_sensors(
        factual,
        prior_evidence_ledger=_empty_ledger(),
        actuator_evidence=evidence,
        actuator_consumption=consumption,
        predicted_actuator_positions_m=predictions,
    )

    with pytest.raises(ValueError, match="differs from the ledger embedded"):
        strict_reweight_factual_intervention_with_independent_sensors(
            first.factual_intervention,
            prior_evidence_ledger=_empty_ledger(),
            actuator_evidence=evidence,
            actuator_consumption=consumption,
            predicted_actuator_positions_m=predictions,
        )


def test_strict_sensor_update_accepts_exact_embedded_ledger_chain() -> None:
    factual = _factual()
    first_evidence = _actuator_evidence()
    first_consumption = evidence_consumption_for_independent_sensor(
        first_evidence,
        source_repository="robot/acquisition",
        source_revision="session-v1",
        correlation_group_id="independent-actuator",
    )
    first_predictions = np.stack(
        (first_evidence.positions_m, first_evidence.positions_m + 0.2),
        axis=0,
    )
    first = strict_reweight_factual_intervention_with_independent_sensors(
        factual,
        prior_evidence_ledger=_empty_ledger(),
        actuator_evidence=first_evidence,
        actuator_consumption=first_consumption,
        predicted_actuator_positions_m=first_predictions,
    )

    second_evidence = replace(
        first_evidence,
        stream_id="measured_end_effector_backup",
        provenance="independent backup robot encoder",
        positions_m=first_evidence.positions_m + 0.05,
    )
    second_consumption = evidence_consumption_for_independent_sensor(
        second_evidence,
        source_repository="robot/acquisition",
        source_revision="session-v1",
        correlation_group_id="independent-actuator-backup",
    )
    second_predictions = np.stack(
        (second_evidence.positions_m, second_evidence.positions_m + 0.1),
        axis=0,
    )
    second = strict_reweight_factual_intervention_with_independent_sensors(
        first.factual_intervention,
        prior_evidence_ledger=first.evidence_ledger,
        actuator_evidence=second_evidence,
        actuator_consumption=second_consumption,
        predicted_actuator_positions_m=second_predictions,
    )

    assert second.sensor_update_applied
    assert {
        entry.consumption_id for entry in second.evidence_ledger.entries
    } == {
        first_consumption.consumption_id,
        second_consumption.consumption_id,
    }
    embedded = second.factual_intervention.metadata["consumed_evidence_ledger"]
    assert embedded["artifact_id"] == second.evidence_ledger.artifact_id
