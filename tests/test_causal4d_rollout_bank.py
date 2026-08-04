import numpy as np
import pytest

from causal4d.rollout_bank import (
    JointRolloutBank,
    PhysicalTrajectoryDistribution,
    SparseTrajectoryEvidence,
)


def _bank() -> JointRolloutBank:
    trajectories = np.zeros((2, 1, 7, 2, 3), dtype=float)
    time = np.arange(7, dtype=float)
    trajectories[0, 0, :, :, 0] = -0.01 * time[:, None]
    trajectories[1, 0, :, :, 0] = 0.01 * time[:, None]
    trajectories[1, 0, :, :, 2] = 0.002 * time[:, None]
    return JointRolloutBank(
        hypothesis_ids=("left", "right"),
        hypothesis_metadata=(
            {"action": {"proposal_id": "left"}},
            {"action": {"proposal_id": "right"}},
        ),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[0.0, 0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=1e-6,
    )


def test_prefix_update_cannot_see_changed_future() -> None:
    bank = _bank()
    observations = bank.trajectories[1, 0].copy()
    changed = observations.copy()
    changed[3:] += 100.0
    first = bank.update_from_observations(
        observations,
        prefix_frame_count=3,
        scale_m=0.005,
        likelihood_power=4.0,
    )
    second = bank.update_from_observations(
        changed,
        prefix_frame_count=3,
        scale_m=0.005,
        likelihood_power=4.0,
    )
    assert np.array_equal(first, second)
    assert first[1, 0] > first[0, 0]


def test_sparse_displacement_evidence_ranks_matching_physical_rollout() -> None:
    bank = _bank()
    nodes = np.asarray([0, 1])
    target = bank.trajectories[1, 0, 1:][:, nodes]
    evidence = SparseTrajectoryEvidence(
        positions_m=target,
        node_indices=nodes,
        rollout_frame_indices=np.arange(1, 7, dtype=float),
        scale_m=0.01,
        likelihood_weight=8.0,
        compare_displacements=True,
        anchor_positions_m=bank.trajectories[1, 0, 0, nodes],
    )
    weights = bank.update_from_sparse_evidence(evidence)
    prediction = bank.predictive_distribution(weights, method="sparse_evidence")
    assert weights[1, 0] > 0.99
    assert prediction.mean.shape == (7, 2, 3)
    assert prediction.interval_lower is not None
    assert np.all(prediction.interval_lower <= prediction.interval_upper)


def test_observation_update_scores_discrepancy_as_readout_not_state() -> None:
    trajectories = np.zeros((1, 2, 5, 1, 3), dtype=float)
    bank = JointRolloutBank(
        hypothesis_ids=("nominal",),
        hypothesis_metadata=({"contact": {}},),
        hypothesis_prior_weights=np.asarray([1.0]),
        parameter_particles=np.asarray([[0.0], [1.0]]),
        parameter_weights=np.asarray([0.5, 0.5]),
        trajectories=trajectories,
    )
    observations = np.zeros((5, 1, 3), dtype=float)
    observations[1:, 0, 0] = 0.01
    discrepancy = np.zeros((2, 1, 3), dtype=float)
    discrepancy[1, 0, 0] = 0.01
    unchanged = bank.trajectories.copy()
    posterior = bank.update_from_observations(
        observations,
        prefix_frame_count=4,
        scale_m=0.001,
        likelihood_power=20.0,
        particle_discrepancy_m=discrepancy,
        particle_discrepancy_variance_m2=np.zeros_like(discrepancy),
    )
    assert posterior[0, 1] > 0.99
    assert np.array_equal(bank.trajectories, unchanged)


def test_rollout_bank_owns_immutable_array_state() -> None:
    trajectories = np.zeros((2, 1, 7, 2, 3), dtype=np.float32)
    hypothesis_weights = np.asarray([0.5, 0.5])
    particles = np.asarray([[0.0, 0.0]])
    parameter_weights = np.asarray([1.0])
    bank = JointRolloutBank(
        hypothesis_ids=("left", "right"),
        hypothesis_metadata=({}, {}),
        hypothesis_prior_weights=hypothesis_weights,
        parameter_particles=particles,
        parameter_weights=parameter_weights,
        trajectories=trajectories,
    )

    trajectories[...] = 1.0
    hypothesis_weights[...] = 0.0
    particles[...] = 2.0
    parameter_weights[...] = 0.0

    assert np.all(bank.trajectories == 0.0)
    assert np.allclose(bank.hypothesis_prior_weights, [0.5, 0.5])
    assert np.all(bank.parameter_particles == 0.0)
    assert np.allclose(bank.parameter_weights, [1.0])
    assert not bank.trajectories.flags.writeable
    assert not bank.hypothesis_prior_weights.flags.writeable
    assert not bank.parameter_particles.flags.writeable
    assert not bank.parameter_weights.flags.writeable


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rollout_bank_rejects_nonfinite_numerical_controls(value: float) -> None:
    with pytest.raises(ValueError, match="variance_floor_m2"):
        JointRolloutBank(
            hypothesis_ids=("only",),
            hypothesis_metadata=({},),
            hypothesis_prior_weights=np.asarray([1.0]),
            parameter_particles=np.asarray([[0.0]]),
            parameter_weights=np.asarray([1.0]),
            trajectories=np.zeros((1, 1, 3, 1, 3)),
            variance_floor_m2=value,
        )

    bank = _bank()
    observations = bank.trajectories[0, 0].copy()
    with pytest.raises(ValueError, match="observation scale"):
        bank.update_from_observations(
            observations,
            prefix_frame_count=3,
            scale_m=value,
        )
    with pytest.raises(ValueError, match="variance_multiplier"):
        bank.predictive_distribution(
            method="unit",
            variance_multiplier=value,
        )
    with pytest.raises(ValueError, match="evidence scale"):
        SparseTrajectoryEvidence(
            positions_m=np.zeros((1, 1, 3)),
            node_indices=np.asarray([0]),
            rollout_frame_indices=np.asarray([0.0]),
            scale_m=value,
            compare_displacements=False,
        )


def test_rollout_bank_marginals_reject_invalid_joint_weights() -> None:
    bank = _bank()
    for invalid in (
        np.zeros((2, 1)),
        np.asarray([[float("nan")], [0.0]]),
        np.asarray([[1.1], [-0.1]]),
    ):
        with pytest.raises(ValueError):
            bank.hypothesis_marginal(invalid)
        with pytest.raises(ValueError):
            bank.parameter_marginal(invalid)


def test_trajectory_distribution_rejects_nonfinite_intervals() -> None:
    mean = np.zeros((2, 1, 3))
    variance = np.ones_like(mean)
    lower = np.zeros_like(mean)
    upper = np.ones_like(mean)
    lower[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="interval bounds must be finite"):
        PhysicalTrajectoryDistribution(
            method="unit",
            mean=mean,
            variance=variance,
            interval_lower=lower,
            interval_upper=upper,
        )


def test_sparse_evidence_rejects_out_of_range_anchor_cleanly() -> None:
    bank = _bank()
    evidence = SparseTrajectoryEvidence(
        positions_m=np.zeros((1, 1, 3)),
        node_indices=np.asarray([0]),
        rollout_frame_indices=np.asarray([0.0]),
        compare_displacements=True,
        anchor_positions_m=np.zeros((1, 3)),
        anchor_rollout_frame=bank.frame_count,
    )
    with pytest.raises(ValueError, match="anchor frame"):
        bank.update_from_sparse_evidence(evidence)


def test_rollout_bank_owns_deeply_immutable_metadata_and_stable_id() -> None:
    metadata = (
        {
            "hypothesis_id": "left",
            "action": {"proposal_id": "left"},
            "nested": {"items": [1, {"accepted": True}]},
        },
        {
            "hypothesis_id": "right",
            "action": {"proposal_id": "right"},
        },
    )
    bank = JointRolloutBank(
        hypothesis_ids=("left", "right"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[0.0, 0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=np.zeros((2, 1, 7, 2, 3), dtype=np.float32),
    )
    artifact_id = bank.artifact_id

    metadata[0]["nested"]["items"][1]["accepted"] = False
    assert bank.hypothesis_metadata[0]["nested"]["items"][1]["accepted"] is True
    assert bank.artifact_id == artifact_id

    with pytest.raises(TypeError, match="immutable"):
        bank.hypothesis_metadata[0]["nested"]["items"].append("mutated")
    with pytest.raises(TypeError, match="immutable"):
        bank.hypothesis_metadata[0]["nested"]["items"][1]["accepted"] = False


def test_rollout_bank_rejects_invalid_or_nonjson_metadata() -> None:
    common = {
        "hypothesis_ids": ("only",),
        "hypothesis_prior_weights": np.asarray([1.0]),
        "parameter_particles": np.asarray([[0.0]]),
        "parameter_weights": np.asarray([1.0]),
        "trajectories": np.zeros((1, 1, 3, 1, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="nonempty string"):
        JointRolloutBank(
            hypothesis_metadata=({"hypothesis_id": 1},),
            **common,
        )
    with pytest.raises(ValueError, match="finite JSON"):
        JointRolloutBank(
            hypothesis_metadata=({"score": float("nan")},),
            **common,
        )
    with pytest.raises(ValueError, match="finite JSON"):
        JointRolloutBank(
            hypothesis_metadata=({1: "coercible-key"},),
            **common,
        )


def test_rollout_bank_artifact_id_covers_metadata_and_arrays() -> None:
    bank = _bank()
    changed_metadata = JointRolloutBank(
        hypothesis_ids=bank.hypothesis_ids,
        hypothesis_metadata=(
            {"action": {"proposal_id": "left"}, "revision": 2},
            bank.hypothesis_metadata[1],
        ),
        hypothesis_prior_weights=bank.hypothesis_prior_weights,
        parameter_particles=bank.parameter_particles,
        parameter_weights=bank.parameter_weights,
        trajectories=bank.trajectories,
        variance_floor_m2=bank.variance_floor_m2,
        confidence_level=bank.confidence_level,
    )
    trajectories = bank.trajectories.copy()
    trajectories[0, 0, 0, 0, 0] = 1.0
    changed_trajectory = JointRolloutBank(
        hypothesis_ids=bank.hypothesis_ids,
        hypothesis_metadata=bank.hypothesis_metadata,
        hypothesis_prior_weights=bank.hypothesis_prior_weights,
        parameter_particles=bank.parameter_particles,
        parameter_weights=bank.parameter_weights,
        trajectories=trajectories,
        variance_floor_m2=bank.variance_floor_m2,
        confidence_level=bank.confidence_level,
    )

    assert bank.artifact_id != changed_metadata.artifact_id
    assert bank.artifact_id != changed_trajectory.artifact_id
