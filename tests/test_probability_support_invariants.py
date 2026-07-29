from __future__ import annotations

import numpy as np
import pytest

from causal4d.baselines import ParameterPosterior
from causal4d.closed_loop import (
    CandidatePlan,
    PlanningConstraints,
    RecedingHorizonPlanner,
    condition_plan_on_recursive_belief,
)
from causal4d.contact_evaluation import _temper_joint_weights
from causal4d.contact_inference import (
    ContactRolloutBank,
    ContactState,
    posterior_predictive_for_state,
)
from causal4d.contracts import (
    FactualIntervention,
    PhysicalPosterior,
    TaskPosterior,
    TwinBelief,
    array_sha256,
    build_causal_context,
)
from causal4d.dynamic_contact import (
    DynamicContactPathBank,
    infer_dynamic_contact_posterior,
)
from causal4d.graph_mode_abduction import (
    GraphModeAbductionConfig,
    graph_mode_joint_weights,
)
from causal4d.hierarchical_abduction import abduct_hierarchical_interventions
from causal4d.molmo_acceptance import _action_ranking
from causal4d.prefix_likelihood import PrefixLikelihoodConfig
from causal4d.rollout_bank import JointRolloutBank
from causal4d.sensor_evidence import ActuatorEvidence
from causal4d.sensor_factorized_abduction import (
    reweight_factual_intervention_with_independent_sensors,
)
from causal4d.simulator import (
    Action,
    GraphObject,
    PhysicalParameters,
    SimulatorConfig,
    simulate_particles,
)


def _context(protocol_id: str = "support_invariants"):
    observations = np.zeros((8, 2, 3), dtype=float)
    actions = np.zeros((8, 1, 3), dtype=float)
    return build_causal_context(
        protocol_id=protocol_id,
        case_id="unit",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=4,
    )


def _physical_plan() -> CandidatePlan:
    context = _context("closed_loop_support")
    trajectories = np.zeros((2, 5, 2, 3), dtype=float)
    trajectories[1, :, :, 0] = np.arange(5, dtype=float)[:, None]
    physical = PhysicalPosterior(
        context=context,
        component_ids=("supported", "excluded"),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((2, 2, 3), 1e-8),
        weights=np.asarray([1.0, 0.0]),
        phi=np.asarray([[1.0], [2.0]]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )
    task = TaskPosterior(
        context=context,
        physical_posterior_id=physical.artifact_id,
        component_ids=physical.component_ids,
        physical_weights=physical.weights,
        task_weights=physical.weights,
        semantic_log_scores=np.asarray([0.0, 1000.0]),
        beta=1.0,
        query_node_indices=np.asarray([0]),
        semantic_source="adversarial unit test",
    )
    return CandidatePlan(
        action_id="plan",
        controller_points_m=np.zeros((4, 1, 3), dtype=float),
        control_anchor_m=np.zeros((1, 3), dtype=float),
        physical=physical,
        task=task,
    )


def _legacy_contact_problem() -> tuple[ContactRolloutBank, SimulatorConfig]:
    graph = GraphObject(
        name="line",
        rest_positions=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        edges=((0, 1),),
        mass=1.0,
        support_stiffness=0.1,
        true_parameters=PhysicalParameters(1.0, 1.0, 1.0),
        sensor_nodes=(0, 1),
    )
    action = Action(
        action_id="push",
        split="test",
        contact_nodes=(0,),
        commanded_forces=np.ones((4, 1, 2), dtype=float),
    )
    states = (
        ContactState((0,), 1.0, 0, 0.0, 0.0),
        ContactState((1,), 1.0, 0, 0.0, 0.0),
    )
    particles = np.asarray([[1.0, 1.0, 1.0], [2.0, 1.0, 2.0]])
    trajectories = np.zeros((2, 2, 5, 2, 2), dtype=float)
    trajectories[1, 1, :, :, 0] = np.arange(5, dtype=float)[:, None]
    bank = ContactRolloutBank(
        graph_object=graph,
        action=action,
        contact_states=states,
        contact_prior_weights=np.asarray([1.0, 0.0]),
        parameter_particles=particles,
        parameter_weights=np.asarray([1.0, 0.0]),
        trajectories=trajectories,
        variance_floor_m2=1e-8,
        confidence_level=0.9,
    )
    return bank, SimulatorConfig(frame_count=5, dt=0.05)


def _graph_mode_problem():
    trajectories = np.zeros((2, 1, 5, 2, 3), dtype=float)
    trajectories[1, 0, :, :, 0] = np.arange(5, dtype=float)[:, None]
    metadata = tuple(
        {
            "action": {
                "proposal_id": "observed",
                "future_action_observed": True,
                "provenance": "unit",
            },
            "contact": {
                "attachment_shifts": [index],
                "gain_multiplier": 1.0,
                "delay_steps": 0,
                "slip_fraction": 0.0,
                "rotation_degrees": 0.0,
            },
        }
        for index in range(2)
    )
    bank = JointRolloutBank(
        hypothesis_ids=("supported", "excluded"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[1.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )
    context = _context("graph_mode_support")
    belief = TwinBelief(
        context=context,
        endpoint_frame=context.o_minus.frame_stop - 1,
        particle_ids=("p0",),
        theta_names=("theta",),
        endpoint_position_m=np.zeros((1, 2, 3)),
        endpoint_velocity_mps=np.zeros((1, 2, 3)),
        theta=np.asarray([[1.0]]),
        discrepancy_mean_m=np.zeros((1, 2, 3)),
        discrepancy_variance_m2=np.zeros((1, 2, 3)),
        weights=np.asarray([1.0]),
    )
    basis = np.asarray([[1.0], [1.0]]) / np.sqrt(2.0)
    return bank, belief, trajectories[1, 0], basis


def _hierarchical_bank() -> JointRolloutBank:
    trajectories = np.zeros((4, 1, 5, 1, 3), dtype=float)
    for index, slope in enumerate((0.0, 1.0, 2.0, 3.0)):
        trajectories[index, 0, :, 0, 0] = slope * np.arange(5, dtype=float)
    metadata = tuple(
        {
            "contact": {
                "gain_multiplier": 0.8 if index < 2 else 1.2,
                "delay_steps": 0,
                "rotation_degrees": 0.0,
                "attachment_shifts": [index % 2],
                "slip_fraction": 0.0,
            }
        }
        for index in range(4)
    )
    return JointRolloutBank(
        hypothesis_ids=("low-a", "low-b", "high-a", "high-b"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.asarray([0.5, 0.0, 0.5, 0.0]),
        parameter_particles=np.asarray([[1.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )


def _factual_with_zero_support() -> FactualIntervention:
    context = _context("sensor_support")
    return FactualIntervention(
        context=context,
        component_ids=("supported", "excluded"),
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        kappa_names=("contact_node", "slip_fraction"),
        phi=np.asarray([[1.0, 0.0, 0.0], [0.8, 1.0, 0.0]]),
        kappa_obs=np.asarray([[0.0, 0.0], [1.0, 0.2]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        weights=np.asarray([1.0, 0.0]),
        evidence_frame_stop=6,
        source_twin_belief_id=array_sha256(np.zeros(1)),
    )


def test_closed_loop_updates_and_semantics_preserve_support() -> None:
    plan = _physical_plan()
    planner = RecedingHorizonPlanner(
        PlanningConstraints(
            maximum_control_step_m=1.0,
            maximum_state_displacement_m=10.0,
            maximum_predictive_std_m=10.0,
        ),
        observation_scale_m=1e-4,
        observation_likelihood_power=100.0,
    )
    assessment = planner.assess(plan)
    assert assessment.semantic_log_evidence == pytest.approx(0.0)

    observations = plan.physical.readout_trajectories_m[1, 1:4].copy()
    belief = planner.assimilate(plan, observations, observation_frame_stop=7)
    np.testing.assert_array_equal(belief.component_weights, [1.0, 0.0])

    rebound = condition_plan_on_recursive_belief(plan, belief)
    np.testing.assert_array_equal(rebound.physical.weights, [1.0, 0.0])
    assert rebound.task is not None
    np.testing.assert_array_equal(rebound.task.task_weights, [1.0, 0.0])


def test_contact_rollout_bank_is_immutable_validated_and_support_preserving() -> None:
    bank, _ = _legacy_contact_problem()
    observations = bank.trajectories[1, 1].copy()
    posterior = bank.update_weights(
        observations,
        prefix_frame_count=4,
        likelihood_scale_m=1e-5,
        likelihood_power=100.0,
    )
    expected = np.zeros((2, 2), dtype=float)
    expected[0, 0] = 1.0
    np.testing.assert_array_equal(posterior, expected)

    no_op = bank.update_weights(
        np.asarray([np.nan]),
        prefix_frame_count=-1,
        likelihood_scale_m=1.0,
        likelihood_power=0.0,
    )
    np.testing.assert_array_equal(no_op, expected)
    assert not bank.trajectories.flags.writeable
    assert not bank.parameter_particles.flags.writeable

    source_particles = bank.parameter_particles.copy()
    source_trajectories = bank.trajectories.copy()
    copied = ContactRolloutBank(
        graph_object=bank.graph_object,
        action=bank.action,
        contact_states=bank.contact_states,
        contact_prior_weights=bank.contact_prior_weights,
        parameter_particles=source_particles,
        parameter_weights=bank.parameter_weights,
        trajectories=source_trajectories,
        variance_floor_m2=bank.variance_floor_m2,
        confidence_level=bank.confidence_level,
    )
    source_particles[:] = 99.0
    source_trajectories[:] = 99.0
    assert not np.any(copied.parameter_particles == 99.0)
    assert not np.any(copied.trajectories == 99.0)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        bank.predictive_distribution(
            np.asarray([[1.2, 0.0], [-0.2, 0.0]]),
            method="invalid",
        )


def test_contact_rollout_bank_rejects_negative_priors() -> None:
    bank, _ = _legacy_contact_problem()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ContactRolloutBank(
            graph_object=bank.graph_object,
            action=bank.action,
            contact_states=bank.contact_states,
            contact_prior_weights=np.asarray([1.2, -0.2]),
            parameter_particles=bank.parameter_particles,
            parameter_weights=bank.parameter_weights,
            trajectories=bank.trajectories,
            variance_floor_m2=bank.variance_floor_m2,
            confidence_level=bank.confidence_level,
        )


def test_fixed_contact_parameter_update_preserves_zero_particle_support() -> None:
    bank, simulator_config = _legacy_contact_problem()
    state = bank.contact_states[0]
    posterior = ParameterPosterior(
        particles=bank.parameter_particles,
        weights=np.asarray([1.0, 0.0]),
        log_likelihood=np.zeros(2),
    )
    trajectories = simulate_particles(
        bank.graph_object,
        state.action(bank.action),
        bank.parameter_particles,
        state.condition(),
        simulator_config,
    )
    result = posterior_predictive_for_state(
        bank.graph_object,
        bank.action,
        state,
        posterior,
        simulator_config=simulator_config,
        variance_floor_m2=1e-8,
        method="support-preserving",
        observations=trajectories[1],
        prefix_frame_count=4,
        likelihood_scale_m=1e-6,
        likelihood_power=100.0,
    )
    np.testing.assert_allclose(result.mean, trajectories[0])


def test_hierarchical_abduction_preserves_local_and_shared_support() -> None:
    bank = _hierarchical_bank()
    result = abduct_hierarchical_interventions(
        [bank],
        [bank.trajectories[3, 0]],
        prefix_frame_counts=[4],
        config=PrefixLikelihoodConfig(
            observation_scale_m=1e-4,
            likelihood_power=100.0,
        ),
        shared_phi_prior=np.asarray([1.0, 0.0]),
    )
    np.testing.assert_array_equal(result.phi_marginal, [1.0, 0.0])
    np.testing.assert_array_equal(result.execution_joint_weights[0][[1, 3]], 0.0)
    assert not result.shared_weights.flags.writeable
    assert not result.execution_joint_weights[0].flags.writeable
    with pytest.raises(TypeError):
        result.metadata["new"] = "mutation"


def test_graph_mode_abduction_preserves_base_support() -> None:
    bank, belief, observations, basis = _graph_mode_problem()
    weights = graph_mode_joint_weights(
        bank,
        belief,
        observations,
        basis,
        prefix_frame_count=4,
        config=GraphModeAbductionConfig(
            position_scale_m=1e-4,
            dynamic_scale_m=1e-4,
            likelihood_temperature=100.0,
        ),
        base_weights=np.asarray([[1.0], [0.0]]),
    )
    np.testing.assert_array_equal(weights[:, 0], [1.0, 0.0])


def test_independent_sensor_abduction_preserves_factual_support() -> None:
    factual = _factual_with_zero_support()
    positions = np.zeros((2, 1, 3), dtype=float)
    evidence = ActuatorEvidence(
        protocol_id=factual.context.protocol_id,
        case_id=factual.context.case_id,
        observed_action_id=factual.context.u_obs.action_id,
        stream_id="encoder",
        clock_id="robot",
        provenance="independent encoder",
        sample_times_s=np.asarray([0.0, 0.1]),
        positions_m=positions,
        variance_m2=np.full_like(positions, 1e-8),
        evidence_frame_stop=factual.evidence_frame_stop,
    )
    predictions = np.stack((positions + 1.0, positions), axis=0)
    updated = reweight_factual_intervention_with_independent_sensors(
        factual,
        actuator_evidence=evidence,
        predicted_actuator_positions_m=predictions,
    )
    np.testing.assert_array_equal(updated.weights, [1.0, 0.0])


def test_dynamic_contact_update_preserves_path_support() -> None:
    trajectories = np.zeros((2, 5, 1, 3), dtype=float)
    trajectories[1, :, 0, 0] = np.arange(5, dtype=float)
    bank = DynamicContactPathBank(
        path_ids=("supported", "excluded"),
        regime_paths=np.asarray([[0] * 5, [1] * 5], dtype=np.int8),
        trajectories_m=trajectories,
        prior_weights=np.asarray([1.0, 0.0]),
        base_variance_m2=1e-8,
    )
    posterior = infer_dynamic_contact_posterior(
        bank,
        trajectories[1],
        prefix_frame_count=4,
        command_activation=np.ones(5),
    )
    np.testing.assert_array_equal(posterior.weights, [1.0, 0.0])


def test_tempering_and_action_ranking_do_not_create_support() -> None:
    np.testing.assert_array_equal(
        _temper_joint_weights(np.asarray([1.0, 0.0]), 100.0),
        [1.0, 0.0],
    )
    metadata = (
        {"action": {"proposal_id": "supported"}},
        {"action": {"proposal_id": "excluded"}},
    )
    bank = JointRolloutBank(
        hypothesis_ids=("supported", "excluded"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.asarray([1.0, 0.0]),
        parameter_particles=np.asarray([[1.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=np.zeros((2, 1, 4, 1, 3)),
    )
    ranking = _action_ranking(
        bank,
        np.asarray([[0.0], [1000.0]]),
        correct_action_id="supported",
        top_k=1,
    )
    assert ranking["top1_action_id"] == "supported"
    assert ranking["action_log_scores"]["excluded"] == -np.inf
