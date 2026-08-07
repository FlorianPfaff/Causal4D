from __future__ import annotations

import numpy as np
import pytest

import causal4d.conditional_uncertainty_v2 as uncertainty_module
from causal4d.baselines import ParameterPosterior
from causal4d.conditional_uncertainty_v2 import (
    ConditionalPredictiveUncertaintyV2,
    joint_predictive_moments_with_conditional_uncertainty_v2,
)
from causal4d.contact_inference import ContactPrior, LatentContactConfig
from causal4d.latent_contact_v2 import (
    ContactV2SupportPolicy,
    GraphContactPatchModelV2,
    build_contact_patch_rollout_bank_v2,
)
from causal4d.simulator import (
    Action,
    GraphObject,
    PhysicalParameters,
    SimulatorConfig,
)


def _bank():
    graph = GraphObject(
        name="guard-chain",
        rest_positions=np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
        edges=((0, 1), (1, 2)),
        mass=1.0,
        support_stiffness=0.2,
        true_parameters=PhysicalParameters(1.0, 0.4, 1.0),
        sensor_nodes=(0, 1, 2),
    )
    force = np.zeros((5, 1, 2), dtype=float)
    force[:, 0, 0] = np.linspace(0.2, 0.6, 5)
    action = Action(
        action_id="push",
        split="test",
        contact_nodes=(1,),
        commanded_forces=force,
    )
    particles = np.asarray(
        ((0.8, 0.3, 0.8), (1.0, 0.4, 1.0), (1.2, 0.5, 1.2)),
        dtype=float,
    )
    weights = np.asarray((0.2, 0.5, 0.3), dtype=float)
    posterior = ParameterPosterior(
        particles=particles,
        weights=weights,
        log_likelihood=np.log(weights),
    )
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
    model = GraphContactPatchModelV2(
        prior=prior,
        config=config,
        patch_spreads=(0.0,),
        patch_spread_probabilities=(1.0,),
        maximum_joint_patches=8,
    )
    return build_contact_patch_rollout_bank_v2(
        graph,
        action,
        posterior,
        model,
        simulator_config=SimulatorConfig(frame_count=action.frame_count, dt=0.03),
        support_policy=ContactV2SupportPolicy(maximum_parameter_count=2),
        variance_floor_m2=1e-6,
        confidence_level=0.9,
    )


def test_dimension_guard_runs_before_quadratic_allocation(monkeypatch) -> None:
    bank = _bank()
    uncertainty = ConditionalPredictiveUncertaintyV2(
        source_artifact_ids=("source",),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("quadratic allocation path was entered")

    monkeypatch.setattr(uncertainty_module.np, "einsum", forbidden)
    with pytest.raises(
        ValueError,
        match="dimension exceeds maximum_joint_dimension",
    ):
        joint_predictive_moments_with_conditional_uncertainty_v2(
            bank,
            uncertainty,
            maximum_joint_dimension=1,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_dimension_guard_requires_a_positive_integer(value) -> None:
    bank = _bank()
    uncertainty = ConditionalPredictiveUncertaintyV2(
        source_artifact_ids=("source",),
    )

    with pytest.raises(ValueError, match="positive integer"):
        joint_predictive_moments_with_conditional_uncertainty_v2(
            bank,
            uncertainty,
            maximum_joint_dimension=value,
        )
