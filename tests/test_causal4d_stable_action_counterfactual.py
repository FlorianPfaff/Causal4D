import numpy as np

from causal4d.action_conditioned_counterfactual import (
    apply_action_conditioned_counterfactual_operator,
)
from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyModel,
)
from causal4d.stable_action_conditioned_counterfactual import (
    apply_stable_action_conditioned_counterfactual_operator,
)
from causal4d.stable_discrepancy_dynamics import (
    StableDiscrepancyTransitionModel,
)
from test_causal4d_action_conditioned_counterfactual import _problem


def _zero_innovation(feature_names: tuple[str, ...]):
    return ActionConditionedDiscrepancyModel(
        feature_names=feature_names,
        base_innovation_covariance_m2=np.zeros((1, 1)),
        feature_directions=np.ones((1, 1)),
        feature_weights=np.zeros((1, len(feature_names))),
    )


def test_identity_transition_matches_persistence_operator() -> None:
    (
        bank,
        manifest,
        twin,
        factual,
        query,
        discrepancy,
        basis,
        anchor,
        feature_names,
    ) = _problem()
    innovation = _zero_innovation(feature_names)
    persistence = apply_action_conditioned_counterfactual_operator(
        bank,
        manifest,
        twin,
        factual,
        query,
        discrepancy,
        innovation,
        basis,
        anchor,
        frame_dt_s=0.1,
    )
    transition = StableDiscrepancyTransitionModel.identity(
        feature_names=feature_names,
        rank=1,
    )
    stable = apply_stable_action_conditioned_counterfactual_operator(
        bank,
        manifest,
        twin,
        factual,
        query,
        discrepancy,
        innovation,
        transition,
        basis,
        anchor,
        frame_dt_s=0.1,
    )

    np.testing.assert_allclose(
        stable.readout_trajectories_m,
        persistence.readout_trajectories_m,
    )
    np.testing.assert_allclose(
        stable.readout_variance_m2,
        persistence.readout_variance_m2,
    )
    np.testing.assert_allclose(
        stable.discrepancy_coefficient_covariance_m2,
        persistence.discrepancy_coefficient_covariance_m2,
    )
    np.testing.assert_allclose(stable.weights, persistence.weights)
    assert stable.metadata["exact_persistence_fallback"] is True
    assert stable.metadata["future_observations_read"] == 0


def test_contraction_changes_counterfactual_readout_only() -> None:
    (
        bank,
        manifest,
        twin,
        factual,
        query,
        discrepancy,
        basis,
        anchor,
        feature_names,
    ) = _problem()
    innovation = _zero_innovation(feature_names)
    contraction_weights = np.zeros((1, len(feature_names)))
    contraction_weights[0, feature_names.index("control_speed_mps")] = 1.0
    transition = StableDiscrepancyTransitionModel(
        feature_names=feature_names,
        rank=1,
        skew_generators=np.zeros((0, 1, 1)),
        skew_feature_weights=np.zeros((0, len(feature_names))),
        contraction_directions=np.ones((1, 1)),
        contraction_feature_weights=contraction_weights,
        drift_directions_m=np.zeros((0, 1, 3)),
        drift_feature_weights=np.zeros((0, len(feature_names))),
    )
    posterior = apply_stable_action_conditioned_counterfactual_operator(
        bank,
        manifest,
        twin,
        factual,
        query,
        discrepancy,
        innovation,
        transition,
        basis,
        anchor,
        frame_dt_s=0.1,
    )

    offset = (
        posterior.readout_trajectories_m
        - posterior.state_trajectories_m
    )[:, :, 0, 0]
    np.testing.assert_allclose(offset[:, 0], 0.002)
    assert np.all(offset[:, -1] < offset[:, 0])
    assert np.all(offset[:, -1] >= 0.0)
    assert posterior.metadata["exact_persistence_fallback"] is False
    np.testing.assert_allclose(
        posterior.state_trajectories_m,
        posterior.physical.state_trajectories_m,
    )
