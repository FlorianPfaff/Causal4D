from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.baselines import ParameterPosterior
from causal4d.contact_inference import ContactPrior, LatentContactConfig
from causal4d.latent_contact_v2 import (
    ContactObservationEvidenceV2,
    ContactV2SupportPolicy,
    ContactV2SupportRejectedError,
    GraphContactPatchModelV2,
    LinearContactObservationGroup,
    SparseContactPatch,
    build_contact_patch_rollout_bank_v2,
    contact_component_log_likelihoods_v2,
    evaluate_contact_v2_support,
    gaussian_mixture_quantiles,
    posterior_weights_from_contact_evidence_v2,
    select_contact_v2_candidate,
)
from causal4d.parameter_support import reduce_parameter_support
from causal4d.simulator import (
    Action,
    GraphObject,
    PhysicalParameters,
    SimulatorConfig,
)


def _graph() -> GraphObject:
    return GraphObject(
        name="chain3",
        rest_positions=np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
        edges=((0, 1), (1, 2)),
        mass=1.0,
        support_stiffness=0.2,
        true_parameters=PhysicalParameters(1.0, 0.4, 1.0),
        sensor_nodes=(0, 1, 2),
    )


def _action(frame_count: int = 6) -> Action:
    force = np.zeros((frame_count - 1, 1, 2), dtype=float)
    force[:, 0, 0] = np.linspace(0.2, 0.6, frame_count - 1)
    return Action(
        action_id="push",
        split="test",
        contact_nodes=(1,),
        commanded_forces=force,
    )


def _posterior() -> ParameterPosterior:
    particles = np.asarray(
        (
            (0.8, 0.3, 0.8),
            (1.0, 0.4, 1.0),
            (1.2, 0.5, 1.2),
            (1.4, 0.6, 1.4),
        ),
        dtype=float,
    )
    weights = np.asarray((0.15, 0.45, 0.30, 0.10), dtype=float)
    return ParameterPosterior(
        particles=particles,
        weights=weights,
        log_likelihood=np.log(weights),
    )


def _contact_model(*, maximum_joint_patches: int = 16) -> GraphContactPatchModelV2:
    config = LatentContactConfig(
        gain_values=(1.0,),
        delay_values=(0,),
        slip_values=(0.0,),
        rotation_values_deg=(0.0,),
        parameter_particle_count=3,
    )
    prior = ContactPrior(
        shift_probability=0.25,
        gain_probabilities=(1.0,),
        delay_probabilities=(1.0,),
        slip_probabilities=(1.0,),
        rotation_probabilities=(1.0,),
        source_objects=("source",),
        source_condition_count=1,
        source_action_split="train",
    )
    return GraphContactPatchModelV2(
        prior=prior,
        config=config,
        patch_spreads=(0.0, 0.5),
        patch_spread_probabilities=(0.7, 0.3),
        maximum_joint_patches=maximum_joint_patches,
    )


def test_dense_builder_includes_endpoint_only_as_difference() -> None:
    observations = np.zeros((5, 2, 2), dtype=float)
    evidence = ContactObservationEvidenceV2.from_dense_prefix(
        observations,
        prefix_frame_count=3,
        position_scale_m=0.1,
        include_positions=False,
        include_differences=True,
    )
    first = evidence.groups[0]
    assert np.any(first.frame_indices == 0)
    for row in np.unique(first.row_indices[first.frame_indices == 0]):
        assert np.sum(first.coefficients[first.row_indices == row]) == pytest.approx(
            0.0
        )


def test_endpoint_cannot_be_used_as_direct_position() -> None:
    with pytest.raises(ValueError, match="zero-sum contrast"):
        LinearContactObservationGroup(
            group_id="bad",
            values_m=np.asarray((0.0,)),
            row_indices=np.asarray((0,)),
            frame_indices=np.asarray((0,)),
            node_indices=np.asarray((0,)),
            coordinate_indices=np.asarray((0,)),
            coefficients=np.asarray((1.0,)),
            covariance_m2=np.asarray(((1.0,),)),
            contributor_ids=("frame0",),
        )


def test_dimension_normalization_removes_coordinate_count_bias() -> None:
    one = LinearContactObservationGroup(
        group_id="one",
        values_m=np.asarray((0.0,)),
        row_indices=np.asarray((0,)),
        frame_indices=np.asarray((1,)),
        node_indices=np.asarray((0,)),
        coordinate_indices=np.asarray((0,)),
        coefficients=np.asarray((1.0,)),
        covariance_m2=np.asarray(((1.0,),)),
        contributor_ids=("one",),
        prior_nominal_probability=0.999999,
        degrees_of_freedom=1e9,
    )
    two = LinearContactObservationGroup(
        group_id="two",
        values_m=np.asarray((0.0, 0.0)),
        row_indices=np.asarray((0, 1)),
        frame_indices=np.asarray((1, 1)),
        node_indices=np.asarray((0, 1)),
        coordinate_indices=np.asarray((0, 0)),
        coefficients=np.asarray((1.0, 1.0)),
        covariance_m2=np.eye(2),
        contributor_ids=("two",),
        prior_nominal_probability=0.999999,
        degrees_of_freedom=1e9,
    )
    components_one = np.zeros((2, 3, 1, 1), dtype=float)
    components_one[1, 1, 0, 0] = 1.0
    components_two = np.zeros((2, 3, 2, 1), dtype=float)
    components_two[1, 1, :, 0] = 1.0
    score_one, _ = contact_component_log_likelihoods_v2(
        components_one,
        ContactObservationEvidenceV2((one,), dimension_normalization_power=1.0),
        prefix_frame_count=2,
    )
    score_two, _ = contact_component_log_likelihoods_v2(
        components_two,
        ContactObservationEvidenceV2((two,), dimension_normalization_power=1.0),
        prefix_frame_count=2,
    )
    assert score_one[0] - score_one[1] == pytest.approx(
        score_two[0] - score_two[1], rel=1e-7, abs=1e-7
    )


def test_full_covariance_changes_contact_evidence() -> None:
    group = LinearContactObservationGroup(
        group_id="correlated",
        values_m=np.zeros(2),
        row_indices=np.asarray((0, 1)),
        frame_indices=np.asarray((1, 1)),
        node_indices=np.asarray((0, 1)),
        coordinate_indices=np.asarray((0, 0)),
        coefficients=np.ones(2),
        covariance_m2=np.asarray(((1.0, 0.9), (0.9, 1.0))),
        contributor_ids=("camera",),
        prior_nominal_probability=0.999,
    )
    evidence = ContactObservationEvidenceV2((group,))
    components = np.zeros((2, 3, 2, 1), dtype=float)
    components[0, 1, :, 0] = (1.0, 1.0)
    components[1, 1, :, 0] = (1.0, -1.0)
    scores, _ = contact_component_log_likelihoods_v2(
        components,
        evidence,
        prefix_frame_count=2,
    )
    assert scores[0] > scores[1]


def test_future_frames_do_not_change_v2_posterior() -> None:
    components = np.zeros((2, 5, 1, 2), dtype=float)
    components[1, 1:3, 0, 0] = 0.5
    observations = np.zeros((5, 1, 2), dtype=float)
    evidence = ContactObservationEvidenceV2.from_dense_prefix(
        observations,
        prefix_frame_count=3,
        position_scale_m=0.1,
    )
    first, _ = posterior_weights_from_contact_evidence_v2(
        np.asarray((0.5, 0.5)),
        components,
        evidence,
        prefix_frame_count=3,
    )
    changed = observations.copy()
    changed[3:] = 1000.0
    changed_evidence = ContactObservationEvidenceV2.from_dense_prefix(
        changed,
        prefix_frame_count=3,
        position_scale_m=0.1,
    )
    second, _ = posterior_weights_from_contact_evidence_v2(
        np.asarray((0.5, 0.5)),
        components,
        changed_evidence,
        prefix_frame_count=3,
    )
    assert np.array_equal(first, second)


def test_sparse_patch_conserves_commanded_force() -> None:
    action = _action()
    patch = SparseContactPatch(
        graph_name="chain3",
        center_nodes=(1,),
        channel_node_weights=np.asarray(((0.25, 0.50, 0.25),)),
    )
    expanded = patch.expanded_action(action)
    assert expanded.contact_nodes == (0, 1, 2)
    assert np.allclose(
        np.sum(expanded.commanded_forces, axis=1),
        np.sum(action.commanded_forces, axis=1),
    )


def test_patch_model_contains_exact_and_distributed_effects() -> None:
    support = _contact_model().hypotheses(_graph(), _action())
    patches = {state.patch.patch_id: state.patch for state in support.states}
    arrays = [patch.channel_node_weights[0] for patch in patches.values()]
    assert any(np.count_nonzero(values) == 1 for values in arrays)
    assert any(np.count_nonzero(values) > 1 for values in arrays)
    assert support.retained_patch_prior_mass == pytest.approx(1.0)


def test_weighted_coreset_represents_full_parameter_mass() -> None:
    posterior = _posterior()
    reduction = reduce_parameter_support(
        posterior.particles,
        posterior.weights,
        maximum_count=2,
        method="weighted_coreset",
    )
    assert reduction.represented_probability_mass == pytest.approx(1.0)
    assert reduction.count == 2


def test_support_policy_rejects_excessive_patch_truncation() -> None:
    posterior = _posterior()
    reduction = reduce_parameter_support(
        posterior.particles,
        posterior.weights,
        maximum_count=2,
        method="weighted_coreset",
    )
    policy = ContactV2SupportPolicy(minimum_retained_patch_prior_mass=0.9)
    decision = evaluate_contact_v2_support(
        reduction,
        retained_patch_prior_mass=0.5,
        policy=policy,
    )
    assert not decision.accepted
    assert decision.reasons == ("patch_prior_mass_not_retained",)


def test_exact_fallback_preserves_object_identity() -> None:
    posterior = _posterior()
    reduction = reduce_parameter_support(
        posterior.particles,
        posterior.weights,
        maximum_count=1,
        method="top_mass",
    )
    decision = evaluate_contact_v2_support(
        reduction,
        retained_patch_prior_mass=0.4,
        policy=ContactV2SupportPolicy(
            parameter_support_method="top_mass",
            maximum_parameter_count=1,
            minimum_represented_parameter_mass=0.9,
            minimum_retained_patch_prior_mass=0.9,
        ),
    )
    baseline = object()
    candidate = object()
    selection = select_contact_v2_candidate(
        decision,
        baseline=baseline,
        candidate=candidate,
    )
    assert selection.deployed is baseline


def test_gaussian_mixture_quantiles_respect_bimodality() -> None:
    means = np.asarray(((-3.0,), (3.0,)))
    variances = np.full_like(means, 0.01)
    quantiles = gaussian_mixture_quantiles(
        means,
        variances,
        np.asarray((0.5, 0.5)),
        (0.05, 0.95),
    )
    assert quantiles[0, 0] == pytest.approx(-3.128, abs=0.01)
    assert quantiles[1, 0] == pytest.approx(3.128, abs=0.01)


def test_build_bank_uses_coreset_and_exact_mixture_intervals() -> None:
    graph = _graph()
    action = _action()
    bank = build_contact_patch_rollout_bank_v2(
        graph,
        action,
        _posterior(),
        _contact_model(),
        simulator_config=SimulatorConfig(frame_count=action.frame_count, dt=0.03),
        support_policy=ContactV2SupportPolicy(
            maximum_parameter_count=2,
            minimum_retained_patch_prior_mass=0.99,
        ),
        variance_floor_m2=1e-6,
        confidence_level=0.9,
    )
    prediction = bank.predictive_distribution()
    assert bank.support_decision.parameter_reduction.method == "weighted_coreset"
    assert prediction.interval_lower is not None
    assert prediction.interval_upper is not None
    assert np.all(prediction.interval_lower <= prediction.interval_upper)


def test_bank_update_and_effect_posterior_are_normalized() -> None:
    graph = _graph()
    action = _action()
    bank = build_contact_patch_rollout_bank_v2(
        graph,
        action,
        _posterior(),
        _contact_model(),
        simulator_config=SimulatorConfig(frame_count=action.frame_count, dt=0.03),
        support_policy=ContactV2SupportPolicy(maximum_parameter_count=2),
        variance_floor_m2=1e-6,
        confidence_level=0.9,
    )
    truth = bank.trajectories_m[0, 0].copy()
    truth[3:] += 999.0
    evidence = ContactObservationEvidenceV2.from_dense_prefix(
        truth,
        prefix_frame_count=3,
        position_scale_m=0.01,
    )
    weights, diagnostics = bank.update_weights(evidence, prefix_frame_count=3)
    effect = bank.effect_posterior(weights)
    assert np.sum(weights) == pytest.approx(1.0)
    assert diagnostics.nominal_responsibilities.shape[-1] == len(evidence.groups)
    assert np.allclose(np.sum(effect.channel_node_mass, axis=1), 1.0)
    assert np.sum(effect.patch_weights) == pytest.approx(1.0)


def test_build_fails_before_simulation_when_support_is_inadmissible() -> None:
    model = replace(_contact_model(maximum_joint_patches=1), maximum_joint_patches=1)
    with pytest.raises(ContactV2SupportRejectedError, match="patch_prior_mass"):
        build_contact_patch_rollout_bank_v2(
            _graph(),
            _action(),
            _posterior(),
            model,
            simulator_config=SimulatorConfig(frame_count=6, dt=0.03),
            support_policy=ContactV2SupportPolicy(
                maximum_parameter_count=2,
                minimum_retained_patch_prior_mass=0.99,
            ),
            variance_floor_m2=1e-6,
            confidence_level=0.9,
        )
