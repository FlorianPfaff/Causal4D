from __future__ import annotations

import numpy as np
import pytest

from causal4d.baselines import ParameterPosterior
from causal4d.contact_inference import (
    ContactState,
    posterior_predictive_for_state,
    select_parameter_support,
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


def test_default_initial_distribution_retains_all_mass() -> None:
    prior = enumerate_contact_paths(np.asarray([0.0]))

    assert prior.path_ids == ("inactive:0-0",)
    assert prior.retained_prior_mass == 1.0


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


def _two_particle_fixed_contact_problem() -> tuple[
    GraphObject,
    Action,
    ContactState,
    ParameterPosterior,
    SimulatorConfig,
]:
    graph, action, state, _, simulator = _fixed_contact_problem()
    posterior = ParameterPosterior(
        particles=np.asarray(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.5],
            ]
        ),
        weights=np.asarray([0.5, 0.5]),
        log_likelihood=np.zeros(2),
    )
    return graph, action, state, posterior, simulator


@pytest.mark.parametrize("maximum_count", [0, -1, True])
def test_parameter_support_rejects_nonpositive_or_boolean_limit(
    maximum_count: int,
) -> None:
    _, _, _, posterior, _ = _two_particle_fixed_contact_problem()

    with pytest.raises(ValueError, match="maximum_count must be a positive integer"):
        select_parameter_support(posterior, maximum_count)


@pytest.mark.parametrize("prefix_frame_count", [True, 0, 1, 5, 6])
def test_fixed_contact_rejects_invalid_prefix_boundaries(
    prefix_frame_count: int,
) -> None:
    graph, action, state, posterior, simulator = _two_particle_fixed_contact_problem()
    observations = np.zeros((5, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="prefix_frame_count"):
        posterior_predictive_for_state(
            graph,
            action,
            state,
            posterior,
            simulator_config=simulator,
            variance_floor_m2=1e-8,
            method="invalid-prefix",
            observations=observations,
            prefix_frame_count=prefix_frame_count,
            likelihood_scale_m=1e-3,
            likelihood_power=1.0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"likelihood_scale_m": 0.0}, "likelihood_scale_m must be positive"),
        ({"likelihood_scale_m": np.nan}, "likelihood_scale_m must be finite"),
        ({"likelihood_power": -1.0}, "likelihood weights must be nonnegative"),
        ({"likelihood_power": np.inf}, "likelihood_power must be finite"),
        (
            {"dynamic_likelihood_weight": -1.0},
            "likelihood weights must be nonnegative",
        ),
        (
            {"dynamic_likelihood_weight": np.nan},
            "dynamic_likelihood_weight must be finite",
        ),
    ],
)
def test_fixed_contact_rejects_invalid_likelihood_settings(
    overrides: dict[str, float],
    message: str,
) -> None:
    graph, action, state, posterior, simulator = _two_particle_fixed_contact_problem()
    observations = np.zeros((5, 2, 2), dtype=float)
    settings = {
        "likelihood_scale_m": 1e-3,
        "likelihood_power": 1.0,
        "dynamic_likelihood_weight": 0.0,
    }
    settings.update(overrides)

    with pytest.raises(ValueError, match=message):
        posterior_predictive_for_state(
            graph,
            action,
            state,
            posterior,
            simulator_config=simulator,
            variance_floor_m2=1e-8,
            method="invalid-likelihood",
            observations=observations,
            prefix_frame_count=3,
            **settings,
        )


def test_fixed_contact_rejects_nonfinite_observation_prefix() -> None:
    graph, action, state, posterior, simulator = _two_particle_fixed_contact_problem()
    observations = np.zeros((5, 2, 2), dtype=float)
    observations[1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="observation prefix must be finite"):
        posterior_predictive_for_state(
            graph,
            action,
            state,
            posterior,
            simulator_config=simulator,
            variance_floor_m2=1e-8,
            method="nonfinite-prefix",
            observations=observations,
            prefix_frame_count=3,
            likelihood_scale_m=1e-3,
            likelihood_power=1.0,
        )


def test_fixed_contact_update_is_invariant_to_future_suffix() -> None:
    graph, action, state, posterior, simulator = _two_particle_fixed_contact_problem()
    observations = np.zeros((5, 2, 2), dtype=float)
    modified = observations.copy()
    modified[3:] = np.nan

    baseline = posterior_predictive_for_state(
        graph,
        action,
        state,
        posterior,
        simulator_config=simulator,
        variance_floor_m2=1e-8,
        method="suffix-invariance",
        observations=observations,
        prefix_frame_count=3,
        likelihood_scale_m=1e-3,
        likelihood_power=1.0,
        dynamic_likelihood_weight=1.0,
    )
    changed = posterior_predictive_for_state(
        graph,
        action,
        state,
        posterior,
        simulator_config=simulator,
        variance_floor_m2=1e-8,
        method="suffix-invariance",
        observations=modified,
        prefix_frame_count=3,
        likelihood_scale_m=1e-3,
        likelihood_power=1.0,
        dynamic_likelihood_weight=1.0,
    )

    np.testing.assert_array_equal(changed.mean, baseline.mean)
    np.testing.assert_array_equal(changed.variance, baseline.variance)
