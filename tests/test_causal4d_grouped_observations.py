import numpy as np
import pytest

from causal4d.grouped_observations import (
    ObservationGroup,
    dense_prefix_observation_groups,
    update_from_grouped_observations,
)
from causal4d.intervention_abduction import FactualAbductionConfig
from causal4d.rollout_bank import JointRolloutBank


def _bank() -> JointRolloutBank:
    trajectories = np.zeros((2, 1, 5, 1, 3), dtype=float)
    trajectories[1, 0, :, 0, 0] = np.arange(5, dtype=float) * 0.01
    return JointRolloutBank(
        hypothesis_ids=("static", "moving"),
        hypothesis_metadata=(
            {"action": {"proposal_id": "static"}},
            {"action": {"proposal_id": "moving"}},
        ),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )


def test_grouped_robust_factor_tolerates_one_gross_outlier() -> None:
    bank = _bank()
    observations = bank.trajectories[1, 0].copy()
    observations[2, 0, 0] = 2.0
    groups = dense_prefix_observation_groups(
        observations,
        prefix_frame_count=4,
        observation_scale_m=0.005,
        likelihood_power=12.0,
        nominal_probability=0.90,
        outlier_scale_multiplier=100.0,
    )
    weights = update_from_grouped_observations(bank, groups)
    assert weights[1, 0] > weights[0, 0]


def test_grouped_update_scores_time_varying_discrepancy() -> None:
    trajectories = np.zeros((1, 2, 4, 1, 3), dtype=float)
    bank = JointRolloutBank(
        hypothesis_ids=("nominal",),
        hypothesis_metadata=({"action": {"proposal_id": "nominal"}},),
        hypothesis_prior_weights=np.asarray([1.0]),
        parameter_particles=np.asarray([[0.0], [1.0]]),
        parameter_weights=np.asarray([0.5, 0.5]),
        trajectories=trajectories,
    )
    observations = np.zeros((4, 1, 3), dtype=float)
    observations[1:, 0, 0] = [0.01, 0.02, 0.03]
    groups = dense_prefix_observation_groups(
        observations,
        prefix_frame_count=4,
        observation_scale_m=0.001,
        likelihood_power=20.0,
        nominal_probability=1.0,
    )
    discrepancy = np.zeros((2, 4, 1, 3), dtype=float)
    discrepancy[1, :, 0, 0] = [0.0, 0.01, 0.02, 0.03]
    weights = update_from_grouped_observations(
        bank,
        groups,
        particle_discrepancy_m=discrepancy,
    )
    assert weights[0, 1] > 0.99


def test_static_discrepancy_uncertainty_cancels_in_increment_factor() -> None:
    bank = _bank()
    group = ObservationGroup(
        frame_index=2,
        reference_frame_index=1,
        node_index=0,
        values_m=np.asarray([0.01, 0.0, 0.0]),
        covariance_m2=np.eye(3) * 1e-6,
        nominal_probability=1.0,
    )
    without_discrepancy = update_from_grouped_observations(bank, (group,))
    static_variance = np.full((1, 1, 3), 100.0)
    with_static_discrepancy = update_from_grouped_observations(
        bank,
        (group,),
        particle_discrepancy_variance_m2=static_variance,
    )
    assert np.allclose(with_static_discrepancy, without_discrepancy)


def test_grouped_covariance_semantics_require_finite_student_t_variance() -> None:
    with pytest.raises(ValueError, match="must exceed two"):
        FactualAbductionConfig(
            observation_model="grouped_robust",
            degrees_of_freedom=2.0,
        )
