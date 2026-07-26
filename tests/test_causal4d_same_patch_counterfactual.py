import numpy as np

from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    TwinBelief,
    build_causal_context,
)
from causal4d.counterfactual import apply_counterfactual_operator
from causal4d.rollout_bank import JointRolloutBank


def _problem(*, same_grasp_semantics: str | None):
    full_frames = 9
    intervention_frame = 4
    observations = np.zeros((full_frames, 1, 3), dtype=float)
    factual_actions = np.zeros((full_frames, 1, 3), dtype=float)
    factual_context = build_causal_context(
        protocol_id="same_patch_counterfactual_unit",
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
        discrepancy_mean_m=np.zeros((1, 1, 3)),
        discrepancy_variance_m2=np.zeros((1, 1, 3)),
        weights=np.asarray([1.0]),
    )
    factual = FactualIntervention(
        context=factual_context,
        component_ids=("patch0::p0", "patch1::p0"),
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        kappa_names=("attachment_shift_hand_0", "slip_fraction"),
        phi=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        kappa_obs=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        hypothesis_indices=np.asarray([0, 2]),
        twin_particle_indices=np.asarray([0, 0]),
        weights=np.asarray([0.1, 0.9]),
        evidence_frame_stop=6,
        source_twin_belief_id=twin.artifact_id,
    )
    counterfactual_actions = factual_actions.copy()
    counterfactual_actions[intervention_frame:, 0, 0] = (
        np.arange(1, 6, dtype=float) * 0.01
    )
    query_context = build_causal_context(
        protocol_id="same_patch_counterfactual_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=factual_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=intervention_frame,
        counterfactual_action_id="new_action",
    )
    query_metadata = (
        {}
        if same_grasp_semantics is None
        else {"same_grasp_semantics": same_grasp_semantics}
    )
    query = CounterfactualQuery(
        context=query_context,
        controller_points_m=counterfactual_actions[intervention_frame:],
        horizon_frames=full_frames - intervention_frame,
        contact_policy="same_grasp",
        source_factual_intervention_id=factual.artifact_id,
        metadata=query_metadata,
    )

    def metadata(identifier: str, shift: int, slip: float):
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
                "slip_fraction": slip,
                "rotation_degrees": 0.0,
            },
        }

    hypotheses = (
        metadata("patch0-slip0", 0, 0.0),
        metadata("patch0-slip5", 0, 0.5),
        metadata("patch1-slip0", 1, 0.0),
        metadata("patch1-slip5", 1, 0.5),
    )
    bank = JointRolloutBank(
        hypothesis_ids=tuple(value["hypothesis_id"] for value in hypotheses),
        hypothesis_metadata=hypotheses,
        hypothesis_prior_weights=np.asarray([0.1, 0.1, 0.2, 0.6]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=np.zeros((4, 1, 6, 1, 3), dtype=np.float32),
    )
    manifest = {
        "causal_context": query_context.as_dict(),
        "twin_belief_id": twin.artifact_id,
    }
    return bank, manifest, twin, factual, query


def test_same_grasp_defaults_to_complete_kappa_reuse() -> None:
    posterior = apply_counterfactual_operator(*_problem(same_grasp_semantics=None))
    np.testing.assert_allclose(posterior.weights, [0.1, 0.0, 0.9, 0.0])
    assert posterior.metadata["factual_kappa_reused"]
    assert posterior.metadata["factual_contact_patch_reused"]
    assert posterior.metadata["factual_slip_reused"]
    assert not posterior.metadata["counterfactual_slip_resampled"]


def test_same_patch_can_resample_slip_under_counterfactual_action() -> None:
    posterior = apply_counterfactual_operator(
        *_problem(same_grasp_semantics="evolve_slip")
    )
    np.testing.assert_allclose(posterior.weights, [0.05, 0.05, 0.225, 0.675])
    assert not posterior.metadata["factual_kappa_reused"]
    assert posterior.metadata["factual_contact_patch_reused"]
    assert not posterior.metadata["factual_slip_reused"]
    assert posterior.metadata["counterfactual_slip_resampled"]
