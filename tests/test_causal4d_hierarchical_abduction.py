import numpy as np

from causal4d.hierarchical_abduction import abduct_hierarchical_interventions
from causal4d.prefix_likelihood import PrefixLikelihoodConfig
from causal4d.rollout_bank import JointRolloutBank


def _bank(offset: float = 0.0) -> JointRolloutBank:
    trajectories = np.zeros((4, 1, 5, 1, 3), dtype=float)
    slopes = [0.0, 0.10, 0.12, 0.24]
    for index, slope in enumerate(slopes):
        trajectories[index, 0, :, 0, 0] = (
            slope * np.arange(5, dtype=float) + offset
        )
    metadata = (
        {
            "contact": {
                "gain_multiplier": 0.8,
                "delay_steps": 0,
                "rotation_degrees": 0,
                "attachment_shifts": [-1],
                "slip_fraction": 0.0,
            }
        },
        {
            "contact": {
                "gain_multiplier": 0.8,
                "delay_steps": 0,
                "rotation_degrees": 0,
                "attachment_shifts": [1],
                "slip_fraction": 0.0,
            }
        },
        {
            "contact": {
                "gain_multiplier": 1.2,
                "delay_steps": 0,
                "rotation_degrees": 0,
                "attachment_shifts": [-1],
                "slip_fraction": 0.0,
            }
        },
        {
            "contact": {
                "gain_multiplier": 1.2,
                "delay_steps": 0,
                "rotation_degrees": 0,
                "attachment_shifts": [1],
                "slip_fraction": 0.0,
            }
        },
    )
    return JointRolloutBank(
        hypothesis_ids=("low-left", "low-right", "high-left", "high-right"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.full(4, 0.25),
        parameter_particles=np.asarray([[1.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )


def test_pooling_sharpens_shared_phi_without_sharing_kappa() -> None:
    bank_a = _bank()
    bank_b = _bank(offset=0.02)
    observation_a = bank_a.trajectories[1, 0].copy()
    observation_a[:, 0, 0] += 0.008 * np.arange(5)
    observation_b = bank_b.trajectories[1, 0].copy()
    observation_b[:, 0, 0] += 0.006 * np.arange(5)
    config = PrefixLikelihoodConfig(
        observation_scale_m=0.05,
        likelihood_power=2.0,
        dynamic_likelihood_weight=0.5,
    )
    single = abduct_hierarchical_interventions(
        [bank_a],
        [observation_a],
        prefix_frame_counts=[4],
        config=config,
    )
    pooled = abduct_hierarchical_interventions(
        [bank_a, bank_b],
        [observation_a, observation_b],
        prefix_frame_counts=[4, 4],
        config=config,
    )
    assert pooled.phi_marginal[0] > single.phi_marginal[0]
    assert pooled.phi_marginal[0] > 0.5
    assert len(pooled.execution_joint_weights) == 2
    for weights in pooled.execution_joint_weights:
        assert np.isclose(weights.sum(), 1.0)
        assert weights.shape == (4, 1)


def test_hierarchical_abduction_is_future_blind_per_execution() -> None:
    bank = _bank()
    observation = bank.trajectories[1, 0].copy()
    changed = observation.copy()
    changed[4:] += 50.0
    config = PrefixLikelihoodConfig(observation_scale_m=0.05)
    first = abduct_hierarchical_interventions(
        [bank],
        [observation],
        prefix_frame_counts=[4],
        config=config,
    )
    second = abduct_hierarchical_interventions(
        [bank],
        [changed],
        prefix_frame_counts=[4],
        config=config,
    )
    assert np.array_equal(first.shared_weights, second.shared_weights)
    assert np.array_equal(
        first.execution_joint_weights[0],
        second.execution_joint_weights[0],
    )
