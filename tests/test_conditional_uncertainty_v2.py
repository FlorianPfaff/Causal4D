from __future__ import annotations

import numpy as np
import pytest

import causal4d.conditional_uncertainty_v2 as uncertainty_module
from causal4d.baselines import ParameterPosterior
from causal4d.conditional_uncertainty_v2 import (
    ConditionalPredictiveUncertaintyV2,
    joint_predictive_moments_with_conditional_uncertainty_v2,
    posterior_weights_with_conditional_uncertainty_v2,
    predictive_distribution_with_conditional_uncertainty_v2,
)
from causal4d.contact_inference import ContactPrior, LatentContactConfig
from causal4d.latent_contact_v2 import (
    ContactObservationEvidenceV2,
    ContactV2SupportPolicy,
    GraphContactPatchModelV2,
    LinearContactObservationGroup,
    build_contact_patch_rollout_bank_v2,
)
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


def _contact_model() -> GraphContactPatchModelV2:
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
        maximum_joint_patches=16,
    )


@pytest.fixture(scope="module")
def bank():
    action = _action()
    return build_contact_patch_rollout_bank_v2(
        _graph(),
        action,
        _posterior(),
        _contact_model(),
        simulator_config=SimulatorConfig(frame_count=action.frame_count, dt=0.03),
        support_policy=ContactV2SupportPolicy(maximum_parameter_count=2),
        variance_floor_m2=1e-6,
        confidence_level=0.9,
    )


def _difference_group() -> LinearContactObservationGroup:
    return LinearContactObservationGroup(
        group_id="difference",
        values_m=np.asarray((0.0,)),
        row_indices=np.asarray((0, 0)),
        frame_indices=np.asarray((0, 1)),
        node_indices=np.asarray((0, 0)),
        coordinate_indices=np.asarray((0, 0)),
        coefficients=np.asarray((-1.0, 1.0)),
        covariance_m2=np.asarray(((1e-4,),)),
        contributor_ids=("camera",),
    )


def test_low_rank_mode_projects_through_linear_group(bank) -> None:
    factors = np.zeros(
        (1, *bank.trajectories_m.shape[-3:]),
        dtype=float,
    )
    factors[0, 0, 0, 0] = 1.0
    factors[0, 1, 0, 0] = 3.0
    uncertainty = ConditionalPredictiveUncertaintyV2(
        source_artifact_ids=("prob4d-mode-belief",),
        low_rank_factors_m=factors,
    )
    evidence = ContactObservationEvidenceV2((_difference_group(),))
    covariance = uncertainty.component_group_covariance_m2(bank, evidence)
    projected = covariance["difference"]
    assert projected.shape == (*bank.prior_joint_weights.shape, 1, 1)
    assert np.allclose(projected[..., 0, 0], 4.0)


def test_joint_covariance_diagonal_matches_marginal_prediction(bank) -> None:
    independent = np.full(bank.trajectories_m.shape[-3:], 2e-6)
    factors = np.zeros(
        (1, *bank.trajectories_m.shape[-3:]),
        dtype=float,
    )
    factors[0, 0, 0, 0] = 1e-3
    factors[0, 0, 0, 1] = 2e-3
    uncertainty = ConditionalPredictiveUncertaintyV2(
        source_artifact_ids=("bpt-conditional-covariance",),
        independent_variance_m2=independent,
        low_rank_factors_m=factors,
    )
    weights = np.zeros_like(bank.prior_joint_weights)
    weights[0, 0] = 1.0
    marginal = predictive_distribution_with_conditional_uncertainty_v2(
        bank,
        uncertainty,
        weights,
        include_intervals=False,
    )
    joint = joint_predictive_moments_with_conditional_uncertainty_v2(
        bank,
        uncertainty,
        weights,
    )
    assert np.allclose(joint.variance_m2, marginal.variance)
    assert joint.covariance_m2[0, 1] == pytest.approx(2e-6)
    assert joint.source_artifact_ids == (bank.bank_id, uncertainty.artifact_id)


def test_weight_update_receives_residual_and_correlated_terms(
    bank,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factors = np.zeros(
        (1, *bank.trajectories_m.shape[-3:]),
        dtype=float,
    )
    factors[0, 0, 0, 0] = 1.0
    factors[0, 1, 0, 0] = 2.0
    uncertainty = ConditionalPredictiveUncertaintyV2(
        source_artifact_ids=("conditional-source",),
        independent_variance_m2=0.5,
        low_rank_factors_m=factors,
    )
    evidence = ContactObservationEvidenceV2((_difference_group(),))
    captured: dict[str, object] = {}

    def capture(
        prior_weights: np.ndarray,
        predicted_components_m: np.ndarray,
        supplied_evidence: ContactObservationEvidenceV2,
        **kwargs: object,
    ) -> tuple[np.ndarray, object]:
        captured.update(kwargs)
        return prior_weights, object()

    monkeypatch.setattr(
        uncertainty_module,
        "posterior_weights_from_contact_evidence_v2",
        capture,
    )
    weights, _ = posterior_weights_with_conditional_uncertainty_v2(
        bank,
        evidence,
        uncertainty,
        prefix_frame_count=2,
    )
    assert np.array_equal(weights, bank.prior_joint_weights)
    assert np.allclose(
        captured["component_variance_m2"],
        0.5 + bank.variance_floor_m2,
    )
    group_covariance = captured["component_group_covariance_m2"]
    assert isinstance(group_covariance, dict)
    assert np.allclose(group_covariance["difference"], 1.0)


def test_uncertainty_identity_binds_provenance_and_arrays_are_immutable() -> None:
    first = ConditionalPredictiveUncertaintyV2(
        source_artifact_ids=("source-a",),
        independent_variance_m2=np.asarray(0.1),
    )
    second = ConditionalPredictiveUncertaintyV2(
        source_artifact_ids=("source-b",),
        independent_variance_m2=np.asarray(0.1),
    )
    assert first.artifact_id != second.artifact_id
    with pytest.raises(ValueError):
        first.independent_variance_m2.setflags(write=True)
