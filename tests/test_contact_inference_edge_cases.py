from __future__ import annotations

import numpy as np
import pytest

from causal4d.baselines import ParameterPosterior
from causal4d.contact_inference import (
    ContactState,
    posterior_predictive_for_state,
)
from causal4d.dynamic_contact import (
    ContactRegime,
    ContactTransitionConfig,
    enumerate_contact_paths,
)
from causal4d.simulator import (
    Action,
    GraphObject,
    PhysicalParameters,
    SimulatorConfig,
)


def test_single_frame_path_reports_initial_pruned_mass() -> None:
    config = ContactTransitionConfig(
        minimum_transition_probability=0.5,
    )
    prior = enumerate_contact_paths(
        np.asarray([0.0]),
        config=config,
        initial_probabilities=np.asarray([0.6, 0.4, 0.0, 0.0]),
    )

    assert prior.path_ids == ("inactive:0-0",)
    assert np.array_equal(
        prior.regime_paths,
        np.asarray([[ContactRegime.INACTIVE]], dtype=np.int8),
    )
    assert np.array_equal(prior.weights, np.ones(1))
    assert prior.retained_prior_mass == pytest.approx(0.6)


def test_initial_path_pruning_fails_with_explicit_error() -> None:
    config = ContactTransitionConfig(
        minimum_transition_probability=0.3,
    )
    with pytest.raises(RuntimeError, match="initial probability mass"):
        enumerate_contact_paths(
            np.asarray([0.0]),
            config=config,
            initial_probabilities=np.full(4, 0.25),
        )


def _fixed_contact_problem() -> tuple[
    GraphObject,
    Action,
    ContactState,
    ParameterPosterior,
    SimulatorConfig,
]:
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
    state = ContactState((0,), 1.0, 0, 0.0, 0.0)
    posterior = ParameterPosterior(
        particles=np.asarray([[1.0, 1.0, 1.0]]),
        weights=np.ones(1),
        log_likelihood=np.zeros(1),
    )
    return (
        graph,
        action,
        state,
        posterior,
        SimulatorConfig(
            frame_count=5,
            dt=0.05,
        ),
    )


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan])
def test_fixed_contact_rejects_invalid_posterior_temperature(
    temperature: float,
) -> None:
    graph, action, state, posterior, simulator = _fixed_contact_problem()
    with pytest.raises(
        ValueError,
        match="posterior_temperature must be finite and positive",
    ):
        posterior_predictive_for_state(
            graph,
            action,
            state,
            posterior,
            simulator_config=simulator,
            variance_floor_m2=1e-8,
            method="invalid-temperature",
            posterior_temperature=temperature,
        )
