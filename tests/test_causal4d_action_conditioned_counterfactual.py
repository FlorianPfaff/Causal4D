import numpy as np

from causal4d.action_conditioned_counterfactual import (
    apply_action_conditioned_counterfactual_operator,
)
from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyModel,
    build_action_conditioned_features,
)
from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    TwinBelief,
    array_sha256,
    build_causal_context,
)
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.rollout_bank import JointRolloutBank
from causal4d.stable_discrepancy_dynamics import (
    StableDiscrepancyTransitionModel,
)


def _problem():
    full_frames = 9
    intervention_frame = 4
    observations = np.zeros((full_frames, 1, 3), dtype=float)
    factual_actions = np.zeros((full_frames, 1, 3), dtype=float)
    factual_context = build_causal_context(
        protocol_id="action_conditioned_counterfactual_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=factual_actions,
        intervention_frame=intervention_frame,
        counterfactual_action_id="factual_action",
    )
    twin = TwinBelief(
        context=factual_context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=("p0",),
        theta_names=("spring",),
        endpoint_position_m=np.zeros((1, 1, 3)),
        endpoint_velocity_mps=np.zeros((1, 1, 3)),
        theta=np.asarray([[0.0]]),
        discrepancy_mean_m=np.asarray([[[0.002, 0.0, 0.0]]]),
        discrepancy_variance_m2=np.full((1, 1, 3), 1e-6),
        weights=np.asarray([1.0]),
    )
    factual = FactualIntervention(
        context=factual_context,
        component_ids=("nominal::p0", "shift::p0"),
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        kappa_names=("attachment_shift_hand_0", "slip_fraction"),
        phi=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        kappa_obs=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        weights=np.asarray([0.25, 0.75]),
        evidence_frame_stop=6,
        source_twin_belief_id=twin.artifact_id,
    )
    counterfactual_actions = factual_actions.copy()
    counterfactual_actions[intervention_frame:, 0, 0] = (
        np.arange(1, 6, dtype=float) * 0.01
    )
    query_context = build_causal_context(
        protocol_id="action_conditioned_counterfactual_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=intervention_frame,
        counterfactual_action_id="new_action",
    )
    query = CounterfactualQuery(
        context=query_context,
        controller_points_m=counterfactual_actions[intervention_frame:],
        horizon_frames=full_frames - intervention_frame,
        contact_policy="same_grasp",
        source_factual_intervention_id=factual.artifact_id,
    )

    def metadata(identifier: str, shift: int):
        return {
            "hypothesis_id": identifier,
            "action": {
                "proposal_id": "new_action",
                "future_action_observed": False,
            },
            "contact": {
                "attachment_shifts": [shift],
                "gain_multiplier": 1.0,
                "delay_steps": 0,
                "slip_fraction": 0.0,
                "rotation_degrees": 0.0,
            },
        }

    trajectories = np.zeros((2, 1, 6, 1, 3), dtype=np.float32)
    trajectories[0, 0, :, 0, 0] = np.arange(6) * 0.01
    trajectories[1, 0, :, 0, 0] = np.arange(6) * 0.012
    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "shift"),
        hypothesis_metadata=(metadata("nominal", 0), metadata("shift", 1)),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=2e-6,
    )
    manifest = {
        "causal_context": query_context.as_dict(),
        "twin_belief_id": twin.artifact_id,
    }
    basis = np.ones((1, 1), dtype=float)
    discrepancy = GraphDiscrepancyBelief(
        basis_sha256=array_sha256(basis),
        component_ids=twin.particle_ids,
        coefficient_mean_m=np.asarray([[[0.002, 0.0, 0.0]]]),
        coefficient_covariance_m2=np.full((1, 3, 1, 1), 1e-6),
        projection_variance_m2=np.zeros(3),
        transition_model_id="persistence",
        innovation_model_id="unit",
    )
    anchor = np.zeros((1, 3), dtype=float)
    feature_names = build_action_conditioned_features(
        query.controller_points_m,
        anchor,
        frame_dt_s=0.1,
        phi_names=factual.phi_names,
        phi=factual.phi[0],
        kappa_names=factual.kappa_names,
        kappa=factual.kappa_obs[0],
    ).names
    return (
        bank,
        manifest,
        twin,
        factual,
        query,
        discrepancy,
        basis,
        anchor,
        feature_names,
    )


def test_zero_innovation_matches_static_readout_moments() -> None:
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
    model = ActionConditionedDiscrepancyModel(
        feature_names=feature_names,
        base_innovation_covariance_m2=np.zeros((1, 1)),
        feature_directions=np.ones((1, 1)),
        feature_weights=np.zeros((1, len(feature_names))),
    )
    posterior = apply_action_conditioned_counterfactual_operator(
        bank,
        manifest,
        twin,
        factual,
        query,
        discrepancy,
        model,
        basis,
        anchor,
        frame_dt_s=0.1,
    )
    offset = posterior.readout_trajectories_m - posterior.state_trajectories_m
    expected_offset = np.broadcast_to(
        np.asarray([0.002, 0.0, 0.0]),
        offset.shape,
    )
    np.testing.assert_allclose(offset, expected_offset)
    np.testing.assert_allclose(posterior.readout_variance_m2, 3e-6)
    assert posterior.component_ids == ("nominal::p0", "shift::p0")
    assert posterior.metadata["future_observations_read"] == 0
    assert posterior.metadata["discrepancy_mean_transition"] == "graph_persistence"


def test_action_conditioned_covariance_grows_over_horizon() -> None:
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
    weights = np.zeros((1, len(feature_names)))
    weights[0, feature_names.index("control_speed_mps")] = 0.02
    model = ActionConditionedDiscrepancyModel(
        feature_names=feature_names,
        base_innovation_covariance_m2=np.asarray([[1e-7]]),
        feature_directions=np.ones((1, 1)),
        feature_weights=weights,
    )
    posterior = apply_action_conditioned_counterfactual_operator(
        bank,
        manifest,
        twin,
        factual,
        query,
        discrepancy,
        model,
        basis,
        anchor,
        frame_dt_s=0.1,
    )
    variance = posterior.readout_variance_m2[:, :, 0, 0]
    assert np.all(np.diff(variance, axis=1) >= -1e-15)
    assert np.all(variance[:, -1] > variance[:, 0])
    offset = posterior.readout_trajectories_m - posterior.state_trajectories_m
    expected_offset = np.broadcast_to(
        np.asarray([0.002, 0.0, 0.0]),
        offset.shape,
    )
    np.testing.assert_allclose(offset, expected_offset)


def test_stable_transition_contracts_discrepancy_mean() -> None:
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
    innovation = ActionConditionedDiscrepancyModel(
        feature_names=feature_names,
        base_innovation_covariance_m2=np.zeros((1, 1)),
        feature_directions=np.zeros((0, 1)),
        feature_weights=np.zeros((0, len(feature_names))),
    )
    contraction_weights = np.zeros((1, len(feature_names)))
    contraction_weights[0, feature_names.index("control_speed_mps")] = 4.0
    transition = StableDiscrepancyTransitionModel(
        feature_names=feature_names,
        rank=1,
        skew_generators=np.zeros((0, 1, 1)),
        skew_feature_weights=np.zeros((0, len(feature_names))),
        contraction_directions=np.ones((1, 1)),
        contraction_feature_weights=contraction_weights,
        drift_directions_m=np.zeros((0, 1, 3)),
        drift_feature_weights=np.zeros((0, len(feature_names))),
        model_id="unit-stable-contraction",
    )
    posterior = apply_action_conditioned_counterfactual_operator(
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
        transition_model=transition,
    )
    offset = (
        posterior.readout_trajectories_m
        - posterior.state_trajectories_m
    )[:, :, 0, 0]
    assert np.all(np.diff(offset, axis=1) <= 1e-15)
    assert np.all(offset[:, -1] < offset[:, 0])
    assert (
        posterior.metadata["discrepancy_mean_transition"]
        == "unit-stable-contraction"
    )
