import numpy as np

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.graph_mode_abduction import (
    GraphModeAbductionConfig,
    abduct_factual_intervention_graph_mode,
    graph_mode_joint_weights,
)
from causal4d.intervention_abduction import factual_joint_weights
from causal4d.rollout_bank import JointRolloutBank


def _problem(node_count: int = 3):
    bank_frames = 8
    trajectories = np.zeros((3, 1, bank_frames, node_count, 3), dtype=float)
    time = np.arange(bank_frames, dtype=float)
    spatial = np.linspace(0.8, 1.2, node_count)
    trajectories[1, 0, :, :, 0] = 0.01 * time[:, None] * spatial[None]
    trajectories[2, 0, :, :, 0] = -0.01 * time[:, None] * spatial[None]

    def metadata(identifier, gain, shift):
        return {
            "hypothesis_id": identifier,
            "action": {
                "proposal_id": "known",
                "future_action_observed": True,
                "provenance": "unit factual action",
            },
            "contact": {
                "attachment_shifts": [shift],
                "gain_multiplier": gain,
                "delay_steps": 0,
                "slip_fraction": 0.0,
                "rotation_degrees": 0.0,
            },
        }

    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "high_gain", "shifted"),
        hypothesis_metadata=(
            metadata("nominal", 1.0, 0),
            metadata("high_gain", 1.15, 0),
            metadata("shifted", 1.0, 1),
        ),
        hypothesis_prior_weights=np.asarray([0.6, 0.2, 0.2]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=1e-8,
    )
    full_frames = 11
    intervention_frame = 4
    full_observations = np.zeros((full_frames, node_count, 3), dtype=float)
    full_observations[intervention_frame - 1 :] = trajectories[1, 0]
    actions = np.zeros((full_frames, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="graph_mode_abduction_unit",
        case_id="synthetic",
        observations=full_observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=intervention_frame,
    )
    belief = TwinBelief(
        context=context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=("p0",),
        theta_names=("spring",),
        endpoint_position_m=np.zeros((1, node_count, 3)),
        endpoint_velocity_mps=np.zeros((1, node_count, 3)),
        theta=np.asarray([[0.0]]),
        discrepancy_mean_m=np.zeros((1, node_count, 3)),
        discrepancy_variance_m2=np.zeros((1, node_count, 3)),
        weights=np.asarray([1.0]),
    )
    observations = trajectories[1, 0].copy()
    mask = np.ones((bank_frames, node_count), dtype=bool)
    basis = np.column_stack(
        (
            np.ones(node_count) / np.sqrt(node_count),
            np.linspace(-1.0, 1.0, node_count),
        )
    )
    basis[:, 1] /= np.linalg.norm(basis[:, 1])
    config = GraphModeAbductionConfig(
        position_scale_m=0.001,
        dynamic_scale_m=0.001,
        dynamic_likelihood_weight=0.5,
        likelihood_temperature=20.0,
        projection_ridge=1e-10,
    )
    return bank, belief, observations, mask, basis, config


def test_graph_mode_abduction_recovers_realized_intervention() -> None:
    bank, belief, observations, mask, basis, config = _problem()
    factual = abduct_factual_intervention_graph_mode(
        bank,
        belief,
        observations,
        basis,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )
    joint = factual_joint_weights(
        factual,
        hypothesis_count=3,
        particle_count=1,
    )
    assert np.argmax(joint[:, 0]) == 1
    assert factual.metadata["endpoint_to_first_o_plus_increment_included"]
    assert factual.metadata["future_frames_read_by_abduction"] == 0
    assert factual.metadata["legacy_factual_abduction_unchanged"]


def test_graph_mode_abduction_is_prefix_only() -> None:
    bank, belief, observations, mask, basis, config = _problem()
    first = graph_mode_joint_weights(
        bank,
        belief,
        observations,
        basis,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )
    changed = observations.copy()
    changed[4:] += 1000.0
    changed_mask = mask.copy()
    changed_mask[4:] = False
    second = graph_mode_joint_weights(
        bank,
        belief,
        changed,
        basis,
        prefix_frame_count=4,
        observation_mask=changed_mask,
        config=config,
    )
    np.testing.assert_array_equal(first, second)


def test_graph_mode_covariance_must_match_basis_rank() -> None:
    bank, belief, observations, mask, basis, _ = _problem()
    config = GraphModeAbductionConfig(mode_covariance_m2=np.eye(3))
    with np.testing.assert_raises_regex(ValueError, "rank differs"):
        graph_mode_joint_weights(
            bank,
            belief,
            observations,
            basis,
            prefix_frame_count=4,
            observation_mask=mask,
            config=config,
        )
