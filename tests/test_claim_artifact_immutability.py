from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from causal4d.closed_loop import CandidatePlan
from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.graph_temporal_discrepancy import GraphTemporalDiscrepancyModel
from causal4d.real_calibration import RealCalibrationCase
from causal4d.semantic_posterior import SparseSemanticEvidence
from causal4d.semantic_trust import SemanticTrustDecision, SemanticValidationCase


_DIGEST = "0" * 64


def _physical_posterior() -> PhysicalPosterior:
    observations = np.zeros((3, 1, 3), dtype=float)
    observed_actions = np.zeros((3, 1, 3), dtype=float)
    counterfactual_actions = np.zeros((3, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="immutability-test",
        case_id="case",
        observations=observations,
        observed_actions=observed_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=1,
    )
    state = np.zeros((1, 3, 1, 3), dtype=np.float32)
    return PhysicalPosterior(
        context=context,
        component_ids=("component",),
        state_trajectories_m=state,
        readout_trajectories_m=state.copy(),
        readout_variance_m2=np.ones((1, 1, 3), dtype=np.float32) * 1e-4,
        weights=np.asarray([1.0]),
        phi=np.zeros((1, 1), dtype=float),
        kappa_cf=np.zeros((1, 1), dtype=float),
        hypothesis_indices=np.asarray([0]),
        twin_particle_indices=np.asarray([0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id=_DIGEST,
        source_factual_intervention_id=_DIGEST,
        source_query_id=_DIGEST,
    )


def _assert_read_only(values: np.ndarray) -> None:
    assert not values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        values[...] = 0
    with pytest.raises(ValueError, match="WRITEABLE"):
        values.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        values.flags.writeable = True


def test_sparse_semantic_evidence_owns_every_retained_array() -> None:
    positions = np.arange(12, dtype=float).reshape(2, 2, 3)
    nodes = np.asarray([0, 1], dtype=np.int64)
    frames = np.asarray([1.0, 2.0])
    anchor = np.arange(6, dtype=float).reshape(2, 3)
    valid = np.ones((2, 2), dtype=bool)
    evidence = SparseSemanticEvidence(
        positions_m=positions,
        node_indices=nodes,
        physical_frame_indices=frames,
        anchor_positions_m=anchor,
        valid=valid,
    )
    expected = (
        evidence.positions_m.copy(),
        evidence.node_indices.copy(),
        evidence.physical_frame_indices.copy(),
        evidence.anchor_positions_m.copy(),
        evidence.valid.copy(),
    )

    positions[...] = -1.0
    nodes[...] = 9
    frames[...] = 9.0
    anchor[...] = -2.0
    valid[...] = False

    actual = (
        evidence.positions_m,
        evidence.node_indices,
        evidence.physical_frame_indices,
        evidence.anchor_positions_m,
        evidence.valid,
    )
    for retained, frozen in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(retained, frozen)
        _assert_read_only(retained)


def test_candidate_plan_owns_control_arrays() -> None:
    physical = _physical_posterior()
    controls = np.arange(6, dtype=float).reshape(2, 1, 3)
    anchor = np.arange(3, dtype=float).reshape(1, 3)
    plan = CandidatePlan(
        action_id="action",
        controller_points_m=controls,
        control_anchor_m=anchor,
        physical=physical,
    )
    expected_controls = controls.copy()
    expected_anchor = anchor.copy()

    controls[...] = -1.0
    anchor[...] = -2.0

    np.testing.assert_array_equal(plan.controller_points_m, expected_controls)
    np.testing.assert_array_equal(plan.control_anchor_m, expected_anchor)
    _assert_read_only(plan.controller_points_m)
    _assert_read_only(plan.control_anchor_m)


def test_real_calibration_case_owns_predictive_arrays() -> None:
    mean = np.zeros((3, 2, 3), dtype=float)
    variance = np.ones_like(mean) * 1e-3
    truth = np.ones_like(mean)
    valid = np.ones((3, 2), dtype=bool)
    case = RealCalibrationCase(
        case_id="case",
        action_id="action",
        contact_region_id="region",
        mean_m=mean,
        variance_m2=variance,
        truth_m=truth,
        valid=valid,
        start_frame=1,
    )
    expected = (
        case.mean_m.copy(),
        case.variance_m2.copy(),
        case.truth_m.copy(),
        case.valid.copy(),
    )

    mean[...] = 4.0
    variance[...] = 5.0
    truth[...] = 6.0
    valid[...] = False

    for retained, frozen in zip(
        (case.mean_m, case.variance_m2, case.truth_m, case.valid),
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(retained, frozen)
        _assert_read_only(retained)


def test_semantic_validation_and_decision_are_deeply_immutable() -> None:
    physical = _physical_posterior()
    evidence = SparseSemanticEvidence(
        positions_m=np.zeros((2, 1, 3), dtype=float),
        node_indices=np.asarray([0]),
        physical_frame_indices=np.asarray([1.0, 2.0]),
        anchor_positions_m=np.zeros((1, 3), dtype=float),
    )
    truth = np.zeros((3, 1, 3), dtype=float)
    mask = np.ones((3, 1), dtype=bool)
    case = SemanticValidationCase(
        case_id="case",
        physical=physical,
        evidence=evidence,
        truth_m=truth,
        mask=mask,
        start_frame=1,
    )
    expected_truth = truth.copy()
    expected_mask = mask.copy()
    truth[...] = 3.0
    mask[...] = False

    np.testing.assert_array_equal(case.truth_m, expected_truth)
    np.testing.assert_array_equal(case.mask, expected_mask)
    _assert_read_only(case.truth_m)
    assert case.mask is not None
    _assert_read_only(case.mask)

    diagnostics = {"support_distance_m": 0.1}
    decision = SemanticTrustDecision(
        calibration_id="calibration",
        selected_beta=1.0,
        applied_beta=1.0,
        accepted=True,
        reasons=(),
        diagnostics=diagnostics,
    )
    diagnostics["support_distance_m"] = 99.0
    assert isinstance(decision.diagnostics, Mapping)
    assert decision.diagnostics["support_distance_m"] == 0.1
    with pytest.raises(TypeError, match="immutable"):
        decision.diagnostics["support_distance_m"] = 1.0


def test_graph_temporal_model_owns_every_retained_array() -> None:
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    eigenvalues = np.asarray([0.0, 1.0])
    transition = np.eye(2) * 0.5
    innovation = np.eye(2) * 0.01
    projection = np.asarray([1e-4, 2e-4, 3e-4])
    model = GraphTemporalDiscrepancyModel(
        basis=basis,
        eigenvalues=eigenvalues,
        transition=transition,
        innovation_covariance=innovation,
        projection_variance_m2=projection,
        selected_rank=2,
        candidate_validation_rmse_m=((1, 0.2), (2, 0.1)),
        spectral_radius_before_clipping=0.5,
        spectral_radius=0.5,
        fit_frame_count=8,
        projection_ridge=1e-5,
        dynamics_ridge=1e-4,
    )
    expected = (
        model.basis.copy(),
        model.eigenvalues.copy(),
        model.transition.copy(),
        model.innovation_covariance.copy(),
        model.projection_variance_m2.copy(),
    )

    basis[...] = -1.0
    eigenvalues[...] = -2.0
    transition[...] = -3.0
    innovation[...] = -4.0
    projection[...] = -5.0

    for retained, frozen in zip(
        (
            model.basis,
            model.eigenvalues,
            model.transition,
            model.innovation_covariance,
            model.projection_variance_m2,
        ),
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(retained, frozen)
        _assert_read_only(retained)
