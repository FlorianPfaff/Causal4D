from __future__ import annotations

import numpy as np
import pytest

from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.grouped_likelihood import posterior_weights_from_grouped_evidence
from causal4d.observation_evidence import GroupedObservationEvidence, ObservationGroup
from causal4d.prefix_likelihood import (
    PrefixLikelihoodConfig,
    update_joint_weights_from_prefix,
)
from causal4d.rollout_bank import JointRolloutBank
from causal4d.semantic_posterior import SparseSemanticEvidence, build_task_posterior
from causal4d.weighting import log_weights_from_probabilities


def _context():
    observations = np.zeros((6, 1, 3), dtype=float)
    actions = np.zeros((6, 1, 3), dtype=float)
    return build_causal_context(
        protocol_id="exact_zero_support",
        case_id="unit",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=2,
    )


def _joint_bank() -> JointRolloutBank:
    trajectories = np.zeros((2, 1, 5, 1, 3), dtype=float)
    trajectories[1, 0, :, 0, 0] = np.arange(5, dtype=float) * 0.1
    return JointRolloutBank(
        hypothesis_ids=("supported", "excluded"),
        hypothesis_metadata=({}, {}),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )


def _physical() -> PhysicalPosterior:
    trajectories = np.zeros((2, 5, 1, 3), dtype=float)
    trajectories[1, :, 0, 0] = np.arange(5, dtype=float) * 0.1
    return PhysicalPosterior(
        context=_context(),
        component_ids=("supported", "excluded"),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((2, 1, 3), 1e-8),
        weights=np.asarray([1.0, 0.0]),
        phi=np.asarray([[1.0], [2.0]]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )


def test_log_weights_preserve_exact_support_and_validate_mass() -> None:
    result = log_weights_from_probabilities(np.asarray([0.75, 0.0, 0.25]))
    assert np.isneginf(result[1])
    np.testing.assert_allclose(np.exp(result[[0, 2]]), [0.75, 0.25])
    with pytest.raises(ValueError, match="positive mass"):
        log_weights_from_probabilities(np.zeros(2))


def test_prefix_update_does_not_resurrect_zero_base_mass() -> None:
    bank = _joint_bank()
    observations = bank.trajectories[1, 0].copy()
    posterior = update_joint_weights_from_prefix(
        bank,
        observations,
        prefix_frame_count=4,
        config=PrefixLikelihoodConfig(
            observation_scale_m=1e-4,
            likelihood_power=100.0,
            dynamic_likelihood_weight=0.0,
        ),
        base_weights=np.asarray([[1.0], [0.0]]),
    )
    np.testing.assert_array_equal(posterior[:, 0], [1.0, 0.0])


def test_grouped_update_does_not_resurrect_zero_prior_mass() -> None:
    bank = _joint_bank()
    evidence = GroupedObservationEvidence(
        groups=(
            ObservationGroup(
                group_id="frame-1",
                values_m=np.asarray([0.1]),
                frame_indices=np.asarray([1]),
                node_indices=np.asarray([0]),
                coordinate_indices=np.asarray([0]),
                covariance_m2=np.asarray([[1e-8]]),
                contributor_ids=("frame-1",),
                prior_nominal_probability=0.95,
                outlier_scale_multiplier=100.0,
                degrees_of_freedom=4.0,
                source_id="unit",
            ),
        )
    )
    posterior, _ = posterior_weights_from_grouped_evidence(
        np.asarray([1.0, 0.0]),
        bank.trajectories[:, 0],
        evidence,
        prefix_frame_count=4,
    )
    np.testing.assert_array_equal(posterior, [1.0, 0.0])


def test_semantic_reweighting_cannot_create_physical_support() -> None:
    physical = _physical()
    evidence = SparseSemanticEvidence(
        positions_m=physical.readout_trajectories_m[1, 1:4],
        node_indices=np.asarray([0]),
        physical_frame_indices=np.asarray([1.0, 2.0, 3.0]),
        scale_m=1e-4,
        compare_displacements=True,
        anchor_positions_m=np.zeros((1, 3)),
    )
    task = build_task_posterior(physical, evidence, beta=100.0)
    np.testing.assert_array_equal(task.task_weights, [1.0, 0.0])
