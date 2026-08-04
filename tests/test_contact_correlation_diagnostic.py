from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from causal4d.benchmark import Episode
from causal4d.contact_concentration_diagnostic import (
    _CalibrationCase,
    scale_probability_weights,
)
from causal4d.contact_correlation_diagnostic import (
    FRAME_BLOCK_POLICY,
    GENERALIZED_BAYES_POLICY,
    NODE_BLOCK_POLICY,
    REGISTERED_POLICY,
    WHITENED_POLICY,
    CorrelationDiagnosticConfig,
    _candidate_descriptor,
    _candidate_weights,
    _decision_rows,
    _graph_distance_node_groups,
    _prepare_case,
    _registered_weights,
    _shrunken_correlation_inverse,
    _source_correlation,
    _temporal_block_energy_m2,
    _whitened_energy_m2,
    write_contact_correlation_diagnostic,
)
from causal4d.contact_evaluation import FoldCalibration
from causal4d.contact_inference import ContactRolloutBank, ContactState
from causal4d.simulator import (
    Action,
    GraphObject,
    PhysicalParameters,
    WorldCondition,
)


def _calibration(*, dynamic_weight: float = 1.0) -> FoldCalibration:
    return FoldCalibration(
        likelihood_scale_m=0.05,
        likelihood_power=0.5,
        dynamic_likelihood_weight=dynamic_weight,
        posterior_temperature=1.5,
        matched_pre_variance_multiplier=1.0,
        shifted_pre_variance_multiplier=1.0,
        matched_online_variance_multiplier=1.0,
        shifted_online_variance_multiplier=1.0,
        source_calibration_rmse_m=0.0,
    )


def _synthetic_case(
    *,
    contact_weights: tuple[float, float] = (0.6, 0.4),
    parameter_weights: tuple[float, float] = (0.7, 0.3),
) -> _CalibrationCase:
    graph_object = GraphObject(
        name="chain",
        rest_positions=np.asarray(((0.0, 0.0), (0.1, 0.0), (0.2, 0.0))),
        edges=((0, 1), (1, 2)),
        mass=1.0,
        support_stiffness=0.2,
        true_parameters=PhysicalParameters(1.0, 1.0, 1.0),
        sensor_nodes=(0, 1, 2),
    )
    action = Action(
        action_id="test",
        split="test",
        contact_nodes=(0,),
        commanded_forces=np.zeros((5, 1, 2), dtype=float),
    )
    states = (
        ContactState((0,), 1.0, 0, 0.0, 0.0),
        ContactState((1,), 1.0, 0, 0.0, 0.0),
    )
    base = np.zeros((6, 3, 2), dtype=float)
    for frame in range(6):
        base[frame, :, 0] = frame * np.asarray((0.010, 0.008, 0.006))
    trajectories = np.empty((2, 2, 6, 3, 2), dtype=float)
    trajectories[0, 0] = base
    trajectories[0, 1] = base + 0.002
    trajectories[1, 0] = base + 0.020
    trajectories[1, 1] = base + 0.024
    bank = ContactRolloutBank(
        graph_object=graph_object,
        action=action,
        contact_states=states,
        contact_prior_weights=np.asarray(contact_weights, dtype=float),
        parameter_particles=np.asarray(((1.0, 1.0, 1.0), (1.1, 1.1, 1.1))),
        parameter_weights=np.asarray(parameter_weights, dtype=float),
        trajectories=trajectories,
        variance_floor_m2=1e-6,
        confidence_level=0.90,
    )
    condition = WorldCondition(name="matched_contact")
    truth = base.copy()
    observations = truth + 0.001
    episode = Episode(
        graph_object=graph_object,
        action=action,
        condition=condition,
        repeat_id=0,
        truth=truth,
        observations=truth.copy(),
        descriptor=np.zeros(7),
    )
    return _CalibrationCase(bank=bank, episode=episode, observations=observations)


def test_registered_energy_reproduces_the_existing_update() -> None:
    case = _synthetic_case()
    calibration = _calibration()
    prepared = _prepare_case(case, calibration, 5)

    actual = _registered_weights(prepared, calibration)
    reference = case.bank.update_weights(
        case.observations,
        prefix_frame_count=5,
        likelihood_scale_m=calibration.likelihood_scale_m,
        likelihood_power=calibration.likelihood_power,
        dynamic_likelihood_weight=calibration.dynamic_likelihood_weight,
    )
    reference = scale_probability_weights(
        reference,
        calibration.posterior_temperature,
    )

    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-15)


def test_all_candidate_likelihoods_preserve_exact_zero_prior_support() -> None:
    case = _synthetic_case(
        contact_weights=(1.0, 0.0),
        parameter_weights=(1.0, 0.0),
    )
    calibration = _calibration()
    prepared = _prepare_case(case, calibration, 5)
    identity = np.eye(prepared.residual_features_m.shape[-1])
    descriptors = (
        _candidate_descriptor(REGISTERED_POLICY, None),
        _candidate_descriptor(FRAME_BLOCK_POLICY, 2),
        _candidate_descriptor(NODE_BLOCK_POLICY, None),
        _candidate_descriptor(WHITENED_POLICY, 0.5),
        _candidate_descriptor(GENERALIZED_BAYES_POLICY, 0.5),
    )

    for descriptor in descriptors:
        weights = _candidate_weights(
            prepared,
            calibration,
            descriptor,
            whitening_inverses={0.5: identity},
        )
        assert np.all(weights[1, :] == 0.0)
        assert np.all(weights[:, 1] == 0.0)
        assert np.isclose(np.sum(weights), 1.0)


def test_temporal_block_size_one_is_the_exact_energy() -> None:
    case = _synthetic_case()
    calibration = _calibration()
    prepared = _prepare_case(case, calibration, 5)

    blocked = _temporal_block_energy_m2(
        prepared.position_residual,
        prepared.velocity_residual,
        prepared.acceleration_residual,
        calibration.dynamic_likelihood_weight,
        1,
    )

    np.testing.assert_allclose(blocked, prepared.exact_energy_m2)


def test_graph_distance_blocks_follow_the_nominal_contact() -> None:
    case = _synthetic_case()

    assert _graph_distance_node_groups(case.bank) == ((0,), (1,), (2,))


def test_identity_whitening_is_the_exact_weighted_residual_energy() -> None:
    case = _synthetic_case()
    calibration = _calibration()
    prepared = _prepare_case(case, calibration, 5)
    identity = np.eye(prepared.residual_features_m.shape[-1])

    whitened = _whitened_energy_m2(prepared.residual_features_m, identity)

    np.testing.assert_allclose(whitened, prepared.exact_energy_m2)


def test_source_correlation_shrinkage_is_positive_definite() -> None:
    matrix = np.asarray(
        (
            (1.0, 0.8, 0.2),
            (0.9, 0.7, 0.3),
            (-1.0, -0.8, -0.1),
            (-0.9, -0.7, -0.2),
        )
    )
    correlation, record = _source_correlation(matrix, variance_floor=1e-6)
    inverse, shrinkage = _shrunken_correlation_inverse(
        correlation,
        0.25,
        eigenvalue_floor=1e-6,
    )

    assert record["source_sample_count"] == 4
    assert shrinkage["shrinkage"] == 0.25
    assert np.all(np.linalg.eigvalsh(inverse) > 0.0)


def test_observation_suffix_does_not_change_prepared_prefix_evidence() -> None:
    case = _synthetic_case()
    calibration = _calibration()
    altered = case.observations.copy()
    altered[5:] = 1e6
    changed = _CalibrationCase(
        bank=case.bank,
        episode=case.episode,
        observations=altered,
    )

    first = _prepare_case(case, calibration, 5)
    second = _prepare_case(changed, calibration, 5)

    np.testing.assert_array_equal(first.exact_energy_m2, second.exact_energy_m2)
    np.testing.assert_array_equal(
        first.residual_features_m,
        second.residual_features_m,
    )


def test_joint_decision_requires_proper_score_and_prediction_preservation() -> None:
    base = {
        "case_count": 10,
        "mean_node_confidence": 0.8,
        "node_calibration_error": 0.0,
        "mean_node_truth_probability": 0.8,
        "mean_node_log_score": 0.2,
        "mean_posterior_entropy_nats": 0.5,
        "mean_posterior_effective_support": 1.7,
        "mean_effective_residual_blocks": 10.0,
    }
    aggregate = [
        {
            **base,
            "policy": REGISTERED_POLICY,
            "world_condition": "matched_contact",
            "node_accuracy": 1.0,
            "mean_node_brier": 0.01,
            "node_credible_coverage": 1.0,
            "mean_trajectory_rmse_m": 0.001,
        },
        {
            **base,
            "policy": REGISTERED_POLICY,
            "world_condition": "shifted_contact",
            "node_accuracy": 0.80,
            "mean_node_brier": 0.30,
            "node_credible_coverage": 0.90,
            "mean_trajectory_rmse_m": 0.001,
        },
        {
            **base,
            "policy": FRAME_BLOCK_POLICY,
            "world_condition": "matched_contact",
            "node_accuracy": 1.0,
            "mean_node_brier": 0.015,
            "node_credible_coverage": 1.0,
            "mean_trajectory_rmse_m": 0.00102,
        },
        {
            **base,
            "policy": FRAME_BLOCK_POLICY,
            "world_condition": "shifted_contact",
            "node_accuracy": 0.80,
            "mean_node_brier": 0.29,
            "node_credible_coverage": 0.90,
            "mean_trajectory_rmse_m": 0.00102,
        },
    ]

    decisions = _decision_rows(aggregate, CorrelationDiagnosticConfig())

    assert decisions == [
        {
            "policy": FRAME_BLOCK_POLICY,
            "shifted_brier_improved": True,
            "matched_brier_preserved": True,
            "node_accuracy_preserved": True,
            "credible_coverage_preserved": True,
            "trajectory_rmse_preserved": True,
            "promotion_candidate": True,
            "interpretation": "candidate_for_new_method_and_new_untouched_panel",
        }
    ]


def test_writer_hashes_every_claim_bearing_payload(tmp_path: Path) -> None:
    result = {
        "schema_version": 1,
        "artifact_kind": "fixture",
        "rows": [{"policy": REGISTERED_POLICY, "node_brier": 0.1}],
        "selection_rows": [{"policy": REGISTERED_POLICY, "candidate": "{}"}],
        "whitening_rows": [{"feature_dimension": 2, "source_sample_count": 4}],
    }

    paths = write_contact_correlation_diagnostic(result, tmp_path)
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))

    assert set(manifest["artifacts"]) == {
        "contact-correlation-diagnostic.json",
        "contact-correlation-rows.csv",
        "contact-correlation-selection.csv",
        "contact-correlation-whitening.csv",
    }
    for name, descriptor in manifest["artifacts"].items():
        payload = (tmp_path / name).read_bytes()
        assert descriptor["bytes"] == len(payload)
        assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
