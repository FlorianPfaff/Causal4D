import numpy as np

from causal4d.prefix_likelihood import (
    PrefixLikelihoodConfig,
    update_joint_weights_from_prefix,
)
from causal4d.rollout_bank import JointRolloutBank


def test_particle_specific_scale_includes_normalization_penalty() -> None:
    trajectories = np.zeros((1, 2, 5, 1, 3), dtype=float)
    bank = JointRolloutBank(
        hypothesis_ids=("nominal",),
        hypothesis_metadata=({},),
        hypothesis_prior_weights=np.asarray([1.0]),
        parameter_particles=np.asarray([[0.0], [1.0]]),
        parameter_weights=np.asarray([0.5, 0.5]),
        trajectories=trajectories,
    )
    discrepancy_variance = np.zeros((2, 1, 3), dtype=float)
    discrepancy_variance[1] = 1.0

    posterior = update_joint_weights_from_prefix(
        bank,
        np.zeros((5, 1, 3), dtype=float),
        prefix_frame_count=4,
        config=PrefixLikelihoodConfig(
            observation_scale_m=0.1,
            position_likelihood_weight=1.0,
            dynamic_likelihood_weight=0.0,
        ),
        particle_discrepancy_variance_m2=discrepancy_variance,
    )

    assert posterior.shape == (1, 2)
    assert posterior[0, 0] > 0.99
    assert posterior[0, 0] > posterior[0, 1]
