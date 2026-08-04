from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.benchmark import Episode
from causal4d.contact_concentration_diagnostic import _CalibrationCase
from causal4d.contact_correlation_diagnostic import (
    REGISTERED_POLICY,
    _prepare_case,
    _registered_weights,
)
from causal4d.contact_evaluation import FoldCalibration
from causal4d.contact_inference import ContactRolloutBank, ContactState
from causal4d.contact_topology_covariance_diagnostic import (
    GLOBAL_POLICY,
    TOPOLOGY_POLICY,
    TopologyCovarianceDiagnosticConfig,
    _candidate_descriptor,
    _candidate_weights,
    _canonical_residual_features,
    _decision_rows,
    _hierarchical_inverse,
    _select_candidate,
    _validate_seed_panel,
    write_contact_topology_covariance_diagnostic,
)
from causal4d.simulator import (
    Action,
    GraphObject,
    PhysicalParameters,
    WorldCondition,
)


def _calibration(dynamic_likelihood_weight: float = 1.0) -> FoldCalibration:
    return FoldCalibration(
        likelihood_scale_m=0.05,
        likelihood_power=0.5,
        dynamic_likelihood_weight=dynamic_likelihood_weight,
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


def test_config_requires_an_exact_noop_candidate() -> None:
    with pytest.raises(ValueError, match="no-op"):
        TopologyCovarianceDiagnosticConfig(identity_shrinkages=(0.25, 0.50))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        TopologyCovarianceDiagnosticConfig(
            shared_correlation_weights=(-0.1, 0.5),
        )


def test_seed_panels_must_be_disjoint() -> None:
    assert _validate_seed_panel((300, 301), (400, 401)) == (
        (300, 301),
        (400, 401),
    )
    with pytest.raises(ValueError, match="disjoint"):
        _validate_seed_panel((300, 301), (301, 400))


def test_hierarchical_covariance_has_shared_and_noop_limits() -> None:
    topology = np.asarray(((1.0, 0.7), (0.7, 1.0)))
    shared = np.asarray(((1.0, -0.2), (-0.2, 1.0)))

    _, topology_record = _hierarchical_inverse(
        topology,
        shared,
        shared_weight=0.0,
        identity_shrinkage=0.25,
        eigenvalue_floor=1e-6,
    )
    np.testing.assert_allclose(
        topology_record["mixed_correlation_matrix"],
        topology,
    )

    _, shared_record = _hierarchical_inverse(
        topology,
        shared,
        shared_weight=1.0,
        identity_shrinkage=0.25,
        eigenvalue_floor=1e-6,
    )
    np.testing.assert_allclose(
        shared_record["mixed_correlation_matrix"],
        shared,
    )

    inverse, noop_record = _hierarchical_inverse(
        topology,
        shared,
        shared_weight=0.5,
        identity_shrinkage=1.0,
        eigenvalue_floor=1e-6,
    )
    np.testing.assert_allclose(inverse, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(
        noop_record["effective_correlation_matrix"],
        np.eye(2),
        atol=1e-14,
    )


def test_identity_whitening_reproduces_registered_weights() -> None:
    case = _synthetic_case()
    calibration = _calibration()
    prepared = _prepare_case(case, calibration, 5)
    descriptor = _candidate_descriptor(
        TOPOLOGY_POLICY,
        shared_weight=0.25,
        identity_shrinkage=1.0,
    )

    actual = _candidate_weights(
        prepared,
        calibration,
        descriptor,
        inverse_correlation=np.eye(prepared.residual_features_m.shape[-1]),
    )
    expected = _registered_weights(prepared, calibration)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_identity_whitening_reproduces_position_only_registered_weights() -> None:
    case = _synthetic_case()
    calibration = _calibration(dynamic_likelihood_weight=0.0)
    prepared = _prepare_case(case, calibration, 5)

    assert prepared.residual_features_m.shape[-1] == 2
    canonical = _canonical_residual_features(prepared.residual_features_m)
    assert canonical.shape[-1] == 6
    np.testing.assert_array_equal(canonical[..., 2:], 0.0)

    actual = _candidate_weights(
        prepared,
        calibration,
        _candidate_descriptor(
            TOPOLOGY_POLICY,
            shared_weight=0.25,
            identity_shrinkage=1.0,
        ),
        inverse_correlation=np.eye(canonical.shape[-1]),
    )
    expected = _registered_weights(prepared, calibration)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_topology_whitening_preserves_exact_zero_prior_support() -> None:
    case = _synthetic_case(
        contact_weights=(1.0, 0.0),
        parameter_weights=(1.0, 0.0),
    )
    calibration = _calibration()
    prepared = _prepare_case(case, calibration, 5)
    inverse = np.eye(prepared.residual_features_m.shape[-1])
    inverse[0, 1] = inverse[1, 0] = 0.2

    weights = _candidate_weights(
        prepared,
        calibration,
        _candidate_descriptor(
            TOPOLOGY_POLICY,
            shared_weight=0.0,
            identity_shrinkage=0.25,
        ),
        inverse_correlation=inverse,
    )

    assert weights[0, 0] == pytest.approx(1.0)
    assert np.count_nonzero(weights) == 1


def test_candidate_selection_uses_brier_then_trajectory_then_order() -> None:
    rows = [
        {
            "candidate_order": 0,
            "candidate": {"policy": TOPOLOGY_POLICY, "value": 0},
            "mean_node_brier": 0.20,
            "mean_trajectory_rmse_m": 0.002,
        },
        {
            "candidate_order": 1,
            "candidate": {"policy": TOPOLOGY_POLICY, "value": 1},
            "mean_node_brier": 0.19,
            "mean_trajectory_rmse_m": 0.003,
        },
        {
            "candidate_order": 2,
            "candidate": {"policy": TOPOLOGY_POLICY, "value": 2},
            "mean_node_brier": 0.19,
            "mean_trajectory_rmse_m": 0.001,
        },
    ]

    assert _select_candidate(rows)["value"] == 2


def _aggregate_row(
    policy: str,
    world: str,
    *,
    brier: float,
    accuracy: float = 0.80,
    coverage: float = 0.90,
    rmse: float = 0.001,
) -> dict[str, object]:
    return {
        "policy": policy,
        "world_condition": world,
        "mean_node_brier": brier,
        "node_accuracy": accuracy,
        "node_credible_coverage": coverage,
        "mean_trajectory_rmse_m": rmse,
    }


def test_topology_decision_requires_global_and_stratum_improvement() -> None:
    aggregate = [
        _aggregate_row(REGISTERED_POLICY, "matched_contact", brier=0.01),
        _aggregate_row(REGISTERED_POLICY, "shifted_contact", brier=0.30),
        _aggregate_row(GLOBAL_POLICY, "matched_contact", brier=0.015),
        _aggregate_row(GLOBAL_POLICY, "shifted_contact", brier=0.296),
        _aggregate_row(TOPOLOGY_POLICY, "matched_contact", brier=0.015),
        _aggregate_row(TOPOLOGY_POLICY, "shifted_contact", brier=0.292),
    ]
    by_topology = []
    for topology, registered, global_value, topology_value in (
        ("rope", 0.20, 0.205, 0.205),
        ("cloth", 0.25, 0.252, 0.252),
        ("soft_block", 0.45, 0.431, 0.419),
    ):
        for policy, value in (
            (REGISTERED_POLICY, registered),
            (GLOBAL_POLICY, global_value),
            (TOPOLOGY_POLICY, topology_value),
        ):
            by_topology.append(
                {
                    "policy": policy,
                    "world_condition": "shifted_contact",
                    "object": topology,
                    "mean_node_brier": value,
                }
            )

    decisions = _decision_rows(
        aggregate,
        by_topology,
        TopologyCovarianceDiagnosticConfig(),
    )
    topology = next(row for row in decisions if row["policy"] == TOPOLOGY_POLICY)

    assert topology["shifted_brier_improved"] is True
    assert topology["shifted_brier_improved_over_global"] is True
    assert topology["per_topology_shifted_brier_preserved"] is True
    assert topology["promotion_candidate"] is True


def test_writer_hashes_every_claim_bearing_payload(tmp_path: Path) -> None:
    result = {
        "schema_version": 1,
        "artifact_kind": "fixture",
        "rows": [{"policy": REGISTERED_POLICY, "node_brier": 0.1}],
        "selection_rows": [
            {
                "policy": TOPOLOGY_POLICY,
                "candidate": {"identity_shrinkage": 1.0},
            }
        ],
        "covariance_rows": [
            {
                "policy": GLOBAL_POLICY,
                "topology": "all",
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
            },
            {
                "policy": TOPOLOGY_POLICY,
                "topology": "soft_block",
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
                "mean_truth_proxy_distance": 0.25,
            },
        ],
    }

    paths = write_contact_topology_covariance_diagnostic(result, tmp_path)
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))

    assert set(manifest["artifacts"]) == {
        "contact-topology-covariance-diagnostic.json",
        "contact-topology-covariance-rows.csv",
        "contact-topology-covariance-selection.csv",
        "contact-topology-covariance-matrices.csv",
    }
    for name, descriptor in manifest["artifacts"].items():
        path = tmp_path / name
        assert descriptor["bytes"] == path.stat().st_size
        assert descriptor["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
