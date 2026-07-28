from __future__ import annotations

import numpy as np

from causal4d.rollout_bank import JointRolloutBank, SparseTrajectoryEvidence


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


def test_dense_update_does_not_resurrect_zero_base_mass() -> None:
    bank = _joint_bank()
    posterior = bank.update_from_observations(
        bank.trajectories[1, 0].copy(),
        prefix_frame_count=4,
        scale_m=1e-4,
        likelihood_power=100.0,
        base_weights=np.asarray([[1.0], [0.0]]),
    )
    np.testing.assert_array_equal(posterior[:, 0], [1.0, 0.0])


def test_sparse_update_does_not_resurrect_zero_base_mass() -> None:
    bank = _joint_bank()
    evidence = SparseTrajectoryEvidence(
        positions_m=bank.trajectories[1, 0, 1:4].copy(),
        node_indices=np.asarray([0]),
        rollout_frame_indices=np.asarray([1.0, 2.0, 3.0]),
        scale_m=1e-4,
        likelihood_weight=100.0,
        compare_displacements=False,
    )
    posterior = bank.update_from_sparse_evidence(
        evidence,
        base_weights=np.asarray([[1.0], [0.0]]),
    )
    np.testing.assert_array_equal(posterior[:, 0], [1.0, 0.0])


def test_zero_weight_sparse_evidence_is_exact_noop() -> None:
    bank = _joint_bank()
    evidence = SparseTrajectoryEvidence(
        positions_m=np.full((1, 1, 3), np.nan),
        node_indices=np.asarray([0]),
        rollout_frame_indices=np.asarray([1.0]),
        likelihood_weight=0.0,
        compare_displacements=False,
        valid=np.zeros((1, 1, 3), dtype=bool),
    )
    base = np.asarray([[0.25], [0.75]])
    posterior = bank.update_from_sparse_evidence(evidence, base_weights=base)
    np.testing.assert_array_equal(posterior, base)
