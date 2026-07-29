from dataclasses import replace

import numpy as np
import pytest

from causal4d.causal_sufficiency import assess_command_residual_sufficiency
from causal4d.contracts import FactualIntervention, array_sha256, build_causal_context
from causal4d.contact_traction import graph_traction_field, integrate_contact_wrench
from causal4d.sensor_evidence import (
    ActuatorEvidence,
    ContactWrenchEvidence,
    load_independent_sensor_evidence,
    save_independent_sensor_evidence,
)
from causal4d.sensor_factorized_abduction import (
    predict_affine_actuator_realizations,
    reweight_factual_intervention_with_independent_sensors,
)


def _factual() -> FactualIntervention:
    observations = np.zeros((8, 2, 3), dtype=float)
    actions = np.zeros((8, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="sensor_factorized_unit",
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


def _sensor_identity() -> dict[str, str]:
    return {
        "protocol_id": "sensor_factorized_unit",
        "case_id": "unit_case",
        "observed_action_id": "u_obs",
    }


def _actuator_evidence(
    *,
    frame_stop: int = 6,
    case_id: str = "unit_case",
    clock_id: str = "robot_monotonic",
) -> ActuatorEvidence:
    positions = np.zeros((2, 1, 3), dtype=float)
    identity = _sensor_identity()
    identity["case_id"] = case_id
    return ActuatorEvidence(
        **identity,
        stream_id="measured_end_effector",
        clock_id=clock_id,
        provenance="robot encoder independent of RGB-D object reconstruction",
        sample_times_s=np.asarray([0.0, 1.0 / 30.0]),
        positions_m=positions,
        variance_m2=np.full_like(positions, 1.0e-4),
        evidence_frame_stop=frame_stop,
    )


def _wrench_evidence(
    *,
    clock_id: str = "robot_monotonic",
) -> ContactWrenchEvidence:
    observed = np.asarray([[1.0, 0.0, 0.0]])
    return ContactWrenchEvidence(
        **_sensor_identity(),
        stream_id="wrist_force",
        clock_id=clock_id,
        provenance="wrist force sensor independent of RGB-D reconstruction",
        sample_times_s=np.asarray([0.0]),
        wrench=observed,
        variance=np.full_like(observed, 1.0e-3),
        quantity_names=("force_x_n", "force_y_n", "force_z_n"),
        evidence_frame_stop=6,
    )


def test_absent_sensor_evidence_returns_same_artifact_exactly() -> None:
    factual = _factual()
    updated = reweight_factual_intervention_with_independent_sensors(factual)
    assert updated is factual


def test_actuator_factor_prefers_matching_realization() -> None:
    factual = _factual()
    evidence = _actuator_evidence()
    matching = evidence.positions_m
    predictions = np.stack((matching, matching + 0.2), axis=0)
    updated = reweight_factual_intervention_with_independent_sensors(
        factual,
        actuator_evidence=evidence,
        predicted_actuator_positions_m=predictions,
    )
    assert updated.weights[0] > 0.999
    diagnostics = updated.metadata["independent_sensor_abduction"]
    assert diagnostics["object_observation_likelihood_reused"] is False
    assert diagnostics["future_object_frames_read"] == 0


def test_actuator_factor_cannot_resurrect_zero_prior_component() -> None:
    factual = replace(_factual(), weights=np.asarray([1.0, 0.0]))
    evidence = _actuator_evidence()
    predictions = np.stack(
        (evidence.positions_m + 0.2, evidence.positions_m),
        axis=0,
    )
    updated = reweight_factual_intervention_with_independent_sensors(
        factual,
        actuator_evidence=evidence,
        predicted_actuator_positions_m=predictions,
    )
    np.testing.assert_array_equal(updated.weights, [1.0, 0.0])


def test_component_invariant_sensor_factor_returns_input_exactly() -> None:
    factual = _factual()
    evidence = _actuator_evidence()
    predictions = np.stack((evidence.positions_m, evidence.positions_m), axis=0)
    updated = reweight_factual_intervention_with_independent_sensors(
        factual,
        actuator_evidence=evidence,
        predicted_actuator_positions_m=predictions,
    )
    assert updated is factual


def test_all_invalid_sensor_samples_return_input_exactly() -> None:
    factual = _factual()
    base = _actuator_evidence()
    evidence = ActuatorEvidence(
        protocol_id=base.protocol_id,
        case_id=base.case_id,
        observed_action_id=base.observed_action_id,
        stream_id=base.stream_id,
        clock_id=base.clock_id,
        provenance=base.provenance,
        sample_times_s=base.sample_times_s,
        positions_m=base.positions_m,
        variance_m2=base.variance_m2,
        evidence_frame_stop=base.evidence_frame_stop,
        valid_mask=np.zeros_like(base.positions_m, dtype=bool),
    )
    predictions = np.zeros((2,) + evidence.positions_m.shape)
    updated = reweight_factual_intervention_with_independent_sensors(
        factual,
        actuator_evidence=evidence,
        predicted_actuator_positions_m=predictions,
    )
    assert updated is factual


def test_wrench_factor_prefers_matching_contact() -> None:
    factual = _factual()
    evidence = _wrench_evidence()
    predictions = np.stack((evidence.wrench, np.zeros_like(evidence.wrench)), axis=0)
    updated = reweight_factual_intervention_with_independent_sensors(
        factual,
        wrench_evidence=evidence,
        predicted_contact_wrench=predictions,
    )
    assert updated.weights[0] > 0.99


def test_sensor_factor_rejects_evidence_beyond_factual_prefix() -> None:
    factual = _factual()
    evidence = _actuator_evidence(frame_stop=7)
    predictions = np.zeros((2,) + evidence.positions_m.shape)
    with pytest.raises(ValueError, match="beyond the factual prefix"):
        reweight_factual_intervention_with_independent_sensors(
            factual,
            actuator_evidence=evidence,
            predicted_actuator_positions_m=predictions,
        )


def test_sensor_factor_rejects_evidence_from_another_case() -> None:
    factual = _factual()
    evidence = _actuator_evidence(case_id="other_case")
    predictions = np.zeros((2,) + evidence.positions_m.shape)
    with pytest.raises(
        ValueError, match="different protocol, case, or observed action"
    ):
        reweight_factual_intervention_with_independent_sensors(
            factual,
            actuator_evidence=evidence,
            predicted_actuator_positions_m=predictions,
        )


def test_sensor_factor_rejects_mismatched_clocks() -> None:
    factual = _factual()
    actuator = _actuator_evidence(clock_id="robot_monotonic")
    wrench = _wrench_evidence(clock_id="force_sensor_clock")
    actuator_predictions = np.zeros((2,) + actuator.positions_m.shape)
    wrench_predictions = np.zeros((2,) + wrench.wrench.shape)
    with pytest.raises(ValueError, match="must use the same clock"):
        reweight_factual_intervention_with_independent_sensors(
            factual,
            actuator_evidence=actuator,
            predicted_actuator_positions_m=actuator_predictions,
            wrench_evidence=wrench,
            predicted_contact_wrench=wrench_predictions,
        )


def test_sensor_evidence_round_trip_is_checksummed(tmp_path) -> None:
    evidence = _actuator_evidence()
    path = tmp_path / "actuator_evidence.npz"
    save_independent_sensor_evidence(path, evidence)
    restored = load_independent_sensor_evidence(path)
    assert type(restored) is ActuatorEvidence
    assert restored.artifact_id == evidence.artifact_id
    assert restored.case_id == evidence.case_id
    assert restored.observed_action_id == evidence.observed_action_id


def test_affine_actuator_model_has_exact_nominal_path_and_delay() -> None:
    commands = np.zeros((3, 1, 3), dtype=float)
    commands[:, 0, 0] = [0.0, 1.0, 2.0]
    phi = np.asarray([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    predicted = predict_affine_actuator_realizations(
        commands,
        ("gain_multiplier", "delay_steps", "rotation_degrees"),
        phi,
    )
    np.testing.assert_allclose(predicted[0], commands)
    np.testing.assert_allclose(predicted[1, :, 0, 0], [0.0, 0.0, 0.5])


def test_graph_traction_integrates_to_expected_wrench() -> None:
    basis = np.eye(2)
    coefficients = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    forces = graph_traction_field(basis, coefficients)
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    wrench = integrate_contact_wrench(positions, forces, np.zeros(3))
    np.testing.assert_allclose(wrench[:3], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(wrench[3:], [0.0, 0.0, 1.0])


def test_sufficiency_test_detects_omitted_command_effect() -> None:
    rng = np.random.default_rng(3)
    count = 60
    realization = rng.normal(size=(count, 2))
    commands = np.asarray(["a", "b"] * (count // 2))
    target = (
        0.2 * realization[:, [0]]
        + 2.0 * (commands == "b")[:, None]
        + rng.normal(scale=0.05, size=(count, 1))
    )
    result = assess_command_residual_sufficiency(
        target,
        realization,
        commands,
        group_ids=[f"execution-{index}" for index in range(count)],
        permutation_count=99,
        random_seed=4,
    )
    assert result.command_effect_detected
    assert result.relative_rmse_reduction > 0.8


def test_sufficiency_test_accepts_command_already_explained_by_realization() -> None:
    rng = np.random.default_rng(5)
    count = 60
    commands = np.asarray(["a", "b"] * (count // 2))
    encoded_command = (commands == "b").astype(float)
    realization = np.column_stack((rng.normal(size=count), encoded_command))
    target = (
        realization[:, [0]]
        + 2.0 * realization[:, [1]]
        + rng.normal(scale=0.05, size=(count, 1))
    )
    result = assess_command_residual_sufficiency(
        target,
        realization,
        commands,
        group_ids=[f"execution-{index}" for index in range(count)],
        permutation_count=49,
        random_seed=6,
    )
    assert not result.command_effect_detected
