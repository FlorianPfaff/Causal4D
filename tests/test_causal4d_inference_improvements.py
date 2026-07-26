import numpy as np

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyModel,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.hierarchical_abduction import abduct_hierarchical_interventions
from causal4d.identifiability import (
    IdentifiabilityConfig,
    assess_intervention_identifiability,
    preserve_prior_within_unidentified_subspace,
)
from causal4d.prefix_likelihood import PrefixLikelihoodConfig
from causal4d.rollout_bank import JointRolloutBank
from causal4d.stable_discrepancy_dynamics import (
    StableDiscrepancyTransitionModel,
    forecast_action_conditioned_dynamics,
)


def _belief(basis: np.ndarray) -> GraphDiscrepancyBelief:
    return GraphDiscrepancyBelief(
        basis_sha256=array_sha256(basis),
        component_ids=("component-0",),
        coefficient_mean_m=np.asarray(
            [[[0.01, 0.0, 0.0], [0.0, -0.02, 0.0]]]
        ),
        coefficient_covariance_m2=np.zeros((1, 3, 2, 2)),
        projection_variance_m2=np.asarray([1e-6, 2e-6, 3e-6]),
        transition_model_id="persistence",
        innovation_model_id="unit-test",
        metadata={"future_frames_read": 0},
    )


def _innovation(
    names: tuple[str, ...],
) -> ActionConditionedDiscrepancyModel:
    return ActionConditionedDiscrepancyModel(
        feature_names=names,
        base_innovation_covariance_m2=np.zeros((2, 2)),
        feature_directions=np.zeros((0, 2)),
        feature_weights=np.zeros((0, len(names))),
    )


def test_identity_transition_preserves_mean_exactly() -> None:
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("speed",),
        values=np.asarray([[0.0], [3.0], [7.0]]),
    )
    transition = StableDiscrepancyTransitionModel.identity(
        feature_names=features.names,
        rank=belief.rank,
    )
    forecast = forecast_action_conditioned_dynamics(
        belief,
        _innovation(features.names),
        transition,
        features,
        basis,
    )
    np.testing.assert_array_equal(
        forecast.coefficient_mean_m,
        np.broadcast_to(
            belief.coefficient_mean_m[:, None],
            forecast.coefficient_mean_m.shape,
        ),
    )


def test_dissipative_transition_rotates_and_contracts_modes() -> None:
    basis = np.eye(2)
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("activation",),
        values=np.asarray([[1.0], [1.0]]),
    )
    transition_model = StableDiscrepancyTransitionModel(
        feature_names=features.names,
        rank=belief.rank,
        skew_generators=np.asarray([[[0.0, -1.0], [1.0, 0.0]]]),
        skew_feature_weights=np.asarray([[0.4]]),
        contraction_directions=np.asarray([[1.0, 0.0]]),
        contraction_feature_weights=np.asarray([[0.5]]),
        drift_directions_m=np.zeros((0, 2, 3)),
        drift_feature_weights=np.zeros((0, 1)),
    )
    operator = transition_model.transition_operator(np.asarray([1.0]))
    assert np.linalg.norm(operator, ord=2) <= 1.0 + 1e-12

    forecast = forecast_action_conditioned_dynamics(
        belief,
        _innovation(features.names),
        transition_model,
        features,
        basis,
    )
    expected = operator @ (operator @ belief.coefficient_mean_m[0])
    np.testing.assert_allclose(forecast.coefficient_mean_m[0, -1], expected)


def test_feature_conditioned_drift_is_norm_capped() -> None:
    model = StableDiscrepancyTransitionModel(
        feature_names=("activation",),
        rank=2,
        skew_generators=np.zeros((0, 2, 2)),
        skew_feature_weights=np.zeros((0, 1)),
        contraction_directions=np.zeros((0, 2)),
        contraction_feature_weights=np.zeros((0, 1)),
        drift_directions_m=np.asarray(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        drift_feature_weights=np.asarray([[1.0]]),
        maximum_drift_norm_m=0.02,
    )
    drift = model.drift_increment_m(np.asarray([10.0]))
    assert np.isclose(np.linalg.norm(drift), 0.02)


def _bank() -> JointRolloutBank:
    trajectories = np.zeros((4, 1, 5, 1, 3), dtype=float)
    slopes = [0.0, 0.10, 0.12, 0.24]
    for index, slope in enumerate(slopes):
        trajectories[index, 0, :, 0, 0] = slope * np.arange(5, dtype=float)
    metadata = tuple(
        {
            "contact": {
                "gain_multiplier": gain,
                "delay_steps": 0,
                "rotation_degrees": 0,
                "attachment_shifts": [shift],
                "slip_fraction": 0.0,
            }
        }
        for gain, shift in ((0.8, -1), (0.8, 1), (1.2, -1), (1.2, 1))
    )
    return JointRolloutBank(
        hypothesis_ids=("low-left", "low-right", "high-left", "high-right"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.full(4, 0.25),
        parameter_particles=np.asarray([[1.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )


def _prefix_config() -> PrefixLikelihoodConfig:
    return PrefixLikelihoodConfig(
        observation_scale_m=0.05,
        likelihood_power=2.0,
        dynamic_likelihood_weight=0.5,
    )


def test_duplicate_same_session_counts_as_one_shared_evidence_unit() -> None:
    bank = _bank()
    observation = bank.trajectories[1, 0].copy()
    observation[:, 0, 0] += 0.008 * np.arange(5)
    single = abduct_hierarchical_interventions(
        [bank],
        [observation],
        prefix_frame_counts=[4],
        config=_prefix_config(),
    )
    duplicate = abduct_hierarchical_interventions(
        [bank, bank],
        [observation, observation],
        prefix_frame_counts=[4, 4],
        config=_prefix_config(),
        session_ids=["session-0", "session-0"],
    )
    np.testing.assert_allclose(single.shared_weights, duplicate.shared_weights)
    assert duplicate.metadata["execution_evidence_powers"] == [0.5, 0.5]
    assert duplicate.metadata["session_count"] == 1


def test_two_independent_sessions_sharpen_shared_phi() -> None:
    bank = _bank()
    observation = bank.trajectories[1, 0].copy()
    observation[:, 0, 0] += 0.008 * np.arange(5)
    single = abduct_hierarchical_interventions(
        [bank],
        [observation],
        prefix_frame_counts=[4],
        config=_prefix_config(),
    )
    independent = abduct_hierarchical_interventions(
        [bank, bank],
        [observation, observation],
        prefix_frame_counts=[4, 4],
        config=_prefix_config(),
        session_ids=["session-0", "session-1"],
    )
    assert independent.phi_marginal[0] > single.phi_marginal[0]


def test_parameter_scaling_makes_unit_conversion_invariant() -> None:
    sensitivity = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    config = IdentifiabilityConfig(
        minimum_information_eigenvalue=1e-8,
        minimum_residualized_response_fraction=0.0,
    )
    reference = assess_intervention_identifiability(
        sensitivity,
        parameter_scale=[1.0, 1.0],
        config=config,
    )
    degrees = sensitivity.copy()
    degrees[:, 1] *= 180.0 / np.pi
    converted = assess_intervention_identifiability(
        degrees,
        parameter_scale=[1.0, np.pi / 180.0],
        config=config,
    )
    np.testing.assert_allclose(
        reference.conditional_information,
        converted.conditional_information,
    )
    np.testing.assert_allclose(reference.eigenvalues, converted.eigenvalues)


def test_partial_update_preserves_prior_inside_unidentified_direction() -> None:
    sensitivity = np.asarray([[1.0, 0.0], [2.0, 0.0]])
    result = assess_intervention_identifiability(
        sensitivity,
        config=IdentifiabilityConfig(
            minimum_residualized_response_fraction=0.0
        ),
    )
    values = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    cleaned = preserve_prior_within_unidentified_subspace(
        np.full(4, 0.25),
        np.asarray([0.05, 0.15, 0.10, 0.70]),
        values,
        result,
    )
    np.testing.assert_allclose(cleaned, [0.10, 0.10, 0.40, 0.40])


def test_rank_zero_partial_update_returns_exact_prior() -> None:
    sensitivity = np.asarray([[1.0], [1.0]])
    result = assess_intervention_identifiability(
        sensitivity,
        sensitivity,
    )
    prior = np.asarray([0.2, 0.8])
    cleaned = preserve_prior_within_unidentified_subspace(
        prior,
        np.asarray([0.9, 0.1]),
        np.asarray([[0.0], [1.0]]),
        result,
    )
    np.testing.assert_array_equal(cleaned, prior)
