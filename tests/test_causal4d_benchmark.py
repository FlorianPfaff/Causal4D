import numpy as np
import pytest

from causal4d.baselines import (
    ParameterPosterior,
    PredictiveDistribution,
    _fit_ridge_trajectory,
    fit_baselines,
)
from causal4d.benchmark import (
    CounterfactualBenchmarkConfig,
    build_protocol,
    generate_episodes,
    make_parameter_grid,
    protocol_manifest,
)


def _config() -> CounterfactualBenchmarkConfig:
    return CounterfactualBenchmarkConfig(
        frame_count=22,
        training_repeats=2,
        parameter_grid_count=3,
        fit_frame_stride=3,
    )


def test_locked_split_repeats_training_actions_and_holds_out_one_action() -> None:
    config = _config()
    protocol = build_protocol(config)[0]
    training, validation, held_out = generate_episodes(protocol, config, seed=7)

    assert len(protocol.train_actions) == 4
    assert len(training) == 4 * config.training_repeats
    assert len(held_out) == 2
    assert {episode.action.action_id for episode in training}.isdisjoint(
        {validation.action.action_id, held_out[0].action.action_id}
    )
    for action in protocol.train_actions:
        repeats = [
            episode
            for episode in training
            if episode.action.action_id == action.action_id
        ]
        assert len(repeats) == config.training_repeats
        assert np.array_equal(repeats[0].descriptor, repeats[1].descriptor)
        assert not np.allclose(repeats[0].truth, repeats[1].truth)


def test_protocol_records_exact_controls_parameters_and_contact_ground_truth() -> None:
    config = _config()
    protocols = build_protocol(config)
    manifest = protocol_manifest(protocols, config)
    first = manifest["objects"][0]

    assert manifest["benchmark"] == "causal4d-controlled-counterfactual-v1"
    assert set(first["object"]["true_parameters"]) == {
        "stiffness",
        "damping",
        "contact_gain",
    }
    assert first["actions"][-1]["split"] == "test"
    assert len(first["actions"][-1]["commanded_forces_n"]) == config.frame_count - 1
    assert manifest["evaluation_rule"]["available_to_models"]
    assert manifest["evaluation_rule"]["evaluator_only"]


def test_parameter_grid_contains_truth_and_fitted_baselines_are_probabilistic() -> None:
    config = _config()
    protocol = build_protocol(config)[1]
    training, validation, held_out = generate_episodes(protocol, config, seed=3)
    particles = make_parameter_grid(protocol.graph_object, config)
    baselines = fit_baselines(training, validation, particles, config)

    assert particles.shape == (config.parameter_grid_count**3, 3)
    assert np.any(
        np.all(
            np.isclose(particles, protocol.graph_object.true_parameters.as_array()),
            axis=1,
        )
    )
    assert np.isclose(np.sum(baselines.physics.posterior.weights), 1.0)
    assert baselines.physics.posterior.effective_sample_size >= 1.0
    assert baselines.hybrid.residual_scale in {0.0, 0.25, 0.5, 0.75, 1.0, 1.25}

    predictions = baselines.predict_all(held_out[0])
    assert [prediction.method for prediction in predictions] == [
        "generative_only",
        "physics_only",
        "hybrid",
    ]
    for prediction in predictions:
        assert prediction.mean.shape == held_out[0].truth.shape
        assert np.all(np.isfinite(prediction.mean))
        assert np.all(prediction.variance > 0.0)


def test_held_out_world_condition_never_changes_model_inputs() -> None:
    config = _config()
    protocol = build_protocol(config)[2]
    training, validation, held_out = generate_episodes(protocol, config, seed=5)
    baselines = fit_baselines(
        training,
        validation,
        make_parameter_grid(protocol.graph_object, config),
        config,
    )
    matched = baselines.predict_all(held_out[0])
    shifted = baselines.predict_all(held_out[1])

    assert not np.allclose(held_out[0].truth, held_out[1].truth)
    for first, second in zip(matched, shifted, strict=True):
        assert np.array_equal(first.mean, second.mean)
        assert np.array_equal(first.variance, second.variance)


def test_baseline_result_arrays_are_owned_and_read_only() -> None:
    particles = np.asarray([[1.0, 2.0, 3.0]])
    weights = np.asarray([1.0])
    log_likelihood = np.asarray([-2.0])
    posterior = ParameterPosterior(particles, weights, log_likelihood)
    particles[...] = 9.0
    weights[...] = 0.0
    log_likelihood[...] = 7.0

    np.testing.assert_array_equal(posterior.particles, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(posterior.weights, [1.0])
    np.testing.assert_array_equal(posterior.log_likelihood, [-2.0])
    assert not posterior.particles.flags.writeable
    assert not posterior.weights.flags.writeable
    assert not posterior.log_likelihood.flags.writeable

    mean = np.zeros((3, 2, 2), dtype=float)
    variance = np.ones_like(mean)
    lower = mean - 1.0
    upper = mean + 1.0
    prediction = PredictiveDistribution(
        method="unit",
        mean=mean,
        variance=variance,
        interval_lower=lower,
        interval_upper=upper,
    )
    mean[...] = 5.0
    variance[...] = 5.0
    lower[...] = 5.0
    upper[...] = 5.0

    assert np.all(prediction.mean == 0.0)
    assert np.all(prediction.variance == 1.0)
    assert np.all(prediction.interval_lower == -1.0)
    assert np.all(prediction.interval_upper == 1.0)
    assert not prediction.mean.flags.writeable
    assert not prediction.variance.flags.writeable
    assert not prediction.interval_lower.flags.writeable
    assert not prediction.interval_upper.flags.writeable


def test_parameter_posterior_rejects_invalid_log_likelihood_support() -> None:
    with pytest.raises(ValueError, match="positive infinity"):
        ParameterPosterior(
            particles=np.zeros((1, 3)),
            weights=np.ones(1),
            log_likelihood=np.asarray([np.inf]),
        )
    with pytest.raises(ValueError, match="zero posterior mass"):
        ParameterPosterior(
            particles=np.zeros((1, 3)),
            weights=np.ones(1),
            log_likelihood=np.asarray([-np.inf]),
        )
    posterior = ParameterPosterior(
        particles=np.zeros((2, 3)),
        weights=np.asarray([1.0, 0.0]),
        log_likelihood=np.asarray([0.0, -np.inf]),
    )
    assert np.isneginf(posterior.log_likelihood[1])


def test_ridge_fit_uses_linear_solves_and_freezes_model_arrays(monkeypatch) -> None:
    def forbidden_inverse(*_args, **_kwargs):
        raise AssertionError("ridge fitting must not form an explicit inverse")

    monkeypatch.setattr(np.linalg, "inv", forbidden_inverse)
    descriptors = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    targets = np.zeros((4, 2, 1, 2), dtype=float)
    targets[:, :, 0, 0] = descriptors[:, :1]
    targets[:, :, 0, 1] = descriptors[:, 1:]
    model = _fit_ridge_trajectory(
        descriptors,
        targets,
        ridge=1e-3,
        variance_floor=1e-8,
    )
    mean, variance = model.predict(np.asarray([0.5, 0.5]))

    assert mean.shape == (2, 1, 2)
    assert variance.shape == mean.shape
    assert model.gram_matrix is not None
    assert model.gram_inverse is None
    for values in (
        model.feature_mean,
        model.feature_scale,
        model.coefficients,
        model.gram_matrix,
        model.residual_variance,
    ):
        assert not values.flags.writeable
