import numpy as np

from causal4d.prefix_likelihood import (
    PrefixLikelihoodConfig,
    prefix_component_log_likelihood,
    update_joint_weights_from_prefix,
)
from causal4d.rollout_bank import JointRolloutBank


def _dynamic_bank() -> JointRolloutBank:
    trajectories = np.zeros((2, 1, 6, 1, 3), dtype=float)
    trajectories[0, 0, :, 0, 0] = [0.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    trajectories[1, 0, :, 0, 0] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    return JointRolloutBank(
        hypothesis_ids=("wrong_first_step", "matching"),
        hypothesis_metadata=({}, {}),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )


def test_dynamic_block_uses_endpoint_to_first_response_increment() -> None:
    bank = _dynamic_bank()
    observations = bank.trajectories[1, 0].copy()
    config = PrefixLikelihoodConfig(
        observation_scale_m=0.1,
        position_likelihood_weight=0.0,
        dynamic_likelihood_weight=1.0,
        likelihood_power=8.0,
    )
    likelihood = prefix_component_log_likelihood(
        bank,
        observations,
        prefix_frame_count=4,
        config=config,
    )
    assert likelihood[1, 0] > likelihood[0, 0]
    posterior = update_joint_weights_from_prefix(
        bank,
        observations,
        prefix_frame_count=4,
        config=config,
    )
    assert posterior[1, 0] > 0.99


def test_prefix_update_is_blind_to_changed_future() -> None:
    bank = _dynamic_bank()
    observations = bank.trajectories[1, 0].copy()
    changed = observations.copy()
    changed[4:] += 100.0
    config = PrefixLikelihoodConfig(observation_scale_m=0.1)
    first = update_joint_weights_from_prefix(
        bank,
        observations,
        prefix_frame_count=4,
        config=config,
    )
    second = update_joint_weights_from_prefix(
        bank,
        changed,
        prefix_frame_count=4,
        config=config,
    )
    assert np.array_equal(first, second)


def test_static_discrepancy_cancels_from_dynamic_block() -> None:
    bank = _dynamic_bank()
    observations = bank.trajectories[1, 0].copy()
    discrepancy = np.asarray([[[0.2, 0.0, 0.0]]])
    config = PrefixLikelihoodConfig(
        observation_scale_m=0.1,
        position_likelihood_weight=0.0,
        dynamic_likelihood_weight=1.0,
    )
    without = prefix_component_log_likelihood(
        bank,
        observations,
        prefix_frame_count=4,
        config=config,
    )
    with_discrepancy = prefix_component_log_likelihood(
        bank,
        observations,
        prefix_frame_count=4,
        config=config,
        particle_discrepancy_m=discrepancy,
    )
    assert np.allclose(without, with_discrepancy)
