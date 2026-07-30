"""Generative-only, physics-only, and hybrid benchmark baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal4d.benchmark import CounterfactualBenchmarkConfig, Episode
from causal4d.immutable_array import readonly_array as _readonly_array
from causal4d.simulator import (
    Action,
    GraphObject,
    WorldCondition,
    simulate_particles,
)


@dataclass(frozen=True)
class ParameterPosterior:
    particles: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray

    def __post_init__(self) -> None:
        particles = _readonly_array(self.particles)
        weights = _readonly_array(self.weights)
        log_likelihood = _readonly_array(self.log_likelihood)
        if particles.ndim != 2 or particles.shape[1] != 3:
            raise ValueError("particles must have shape (particle, 3)")
        if weights.shape != (particles.shape[0],):
            raise ValueError("weights must match particle count")
        if log_likelihood.shape != weights.shape:
            raise ValueError("log_likelihood must match particle count")
        if not np.all(np.isfinite(particles)) or not np.all(np.isfinite(weights)):
            raise ValueError("posterior particles and weights must be finite")
        if np.any(np.isnan(log_likelihood)) or np.any(np.isposinf(log_likelihood)):
            raise ValueError("log_likelihood must not contain NaN or positive infinity")
        if np.any(np.isneginf(log_likelihood) & (weights > 0.0)):
            raise ValueError(
                "negative-infinite likelihood requires zero posterior mass"
            )
        if np.any(weights < 0.0) or not np.isclose(
            np.sum(weights),
            1.0,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("weights must be non-negative and sum to one")
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "log_likelihood", log_likelihood)

    @property
    def mean(self) -> np.ndarray:
        return np.sum(self.weights[:, None] * self.particles, axis=0)

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(np.square(self.weights)))


@dataclass(frozen=True)
class PredictiveDistribution:
    method: str
    mean: np.ndarray
    variance: np.ndarray
    interval_lower: np.ndarray | None = None
    interval_upper: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("predictive method must be nonempty")
        mean = _readonly_array(self.mean)
        variance = _readonly_array(self.variance)
        if mean.ndim != 3 or mean.shape[-1] != 2:
            raise ValueError("predictive mean must have shape (frame, node, 2)")
        if variance.shape != mean.shape:
            raise ValueError("predictive variance must match the mean")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("predictive arrays must be finite")
        if np.any(variance <= 0.0):
            raise ValueError("predictive variances must be positive")
        if (self.interval_lower is None) != (self.interval_upper is None):
            raise ValueError("predictive interval bounds must be provided together")
        if self.interval_lower is not None and self.interval_upper is not None:
            lower = _readonly_array(self.interval_lower)
            upper = _readonly_array(self.interval_upper)
            if lower.shape != mean.shape or upper.shape != mean.shape:
                raise ValueError("predictive interval bounds must match the mean")
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                raise ValueError("predictive interval bounds must be finite")
            if np.any(lower > upper):
                raise ValueError("predictive interval lower bound exceeds upper bound")
            object.__setattr__(self, "interval_lower", lower)
            object.__setattr__(self, "interval_upper", upper)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)


@dataclass(frozen=True)
class RidgeTrajectoryModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    gram_inverse: np.ndarray | None
    residual_variance: np.ndarray
    output_shape: tuple[int, int, int]
    gram_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        feature_mean = _readonly_array(self.feature_mean)
        feature_scale = _readonly_array(self.feature_scale)
        coefficients = _readonly_array(self.coefficients)
        residual_variance = _readonly_array(self.residual_variance)
        if feature_mean.ndim != 1 or feature_scale.shape != feature_mean.shape:
            raise ValueError("feature statistics must be aligned vectors")
        if not np.all(np.isfinite(feature_mean)) or not np.all(
            np.isfinite(feature_scale)
        ):
            raise ValueError("feature statistics must be finite")
        if np.any(feature_scale <= 0.0):
            raise ValueError("feature scales must be positive")
        feature_count = len(feature_mean) + 1
        output_size = int(np.prod(self.output_shape))
        if coefficients.shape != (feature_count, output_size):
            raise ValueError("coefficients do not match features and output shape")
        if residual_variance.shape != (output_size,):
            raise ValueError("residual_variance does not match output shape")
        if not np.all(np.isfinite(coefficients)) or not np.all(
            np.isfinite(residual_variance)
        ):
            raise ValueError("ridge model arrays must be finite")
        if np.any(residual_variance <= 0.0):
            raise ValueError("residual variances must be positive")
        if (self.gram_inverse is None) == (self.gram_matrix is None):
            raise ValueError("provide exactly one of gram_inverse or gram_matrix")
        gram_inverse = None
        gram_matrix = None
        if self.gram_inverse is not None:
            gram_inverse = _readonly_array(self.gram_inverse)
            if gram_inverse.shape != (feature_count, feature_count):
                raise ValueError("gram_inverse has incompatible shape")
            if not np.all(np.isfinite(gram_inverse)):
                raise ValueError("gram_inverse must be finite")
        if self.gram_matrix is not None:
            gram_matrix = _readonly_array(self.gram_matrix)
            if gram_matrix.shape != (feature_count, feature_count):
                raise ValueError("gram_matrix has incompatible shape")
            if not np.all(np.isfinite(gram_matrix)):
                raise ValueError("gram_matrix must be finite")
            if not np.allclose(gram_matrix, gram_matrix.T, atol=1e-12, rtol=1e-12):
                raise ValueError("gram_matrix must be symmetric")
            try:
                np.linalg.cholesky(gram_matrix)
            except np.linalg.LinAlgError as error:
                raise ValueError("gram_matrix must be positive definite") from error
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "gram_inverse", gram_inverse)
        object.__setattr__(self, "gram_matrix", gram_matrix)
        object.__setattr__(self, "residual_variance", residual_variance)

    def predict(self, descriptor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        descriptor = np.asarray(descriptor, dtype=float)
        if descriptor.shape != self.feature_mean.shape:
            raise ValueError("descriptor shape does not match fitted features")
        if not np.all(np.isfinite(descriptor)):
            raise ValueError("descriptor must be finite")
        standardized = (descriptor - self.feature_mean) / self.feature_scale
        feature = np.concatenate(([1.0], standardized))
        mean = feature @ self.coefficients
        if self.gram_matrix is not None:
            leverage_vector = np.linalg.solve(self.gram_matrix, feature)
            leverage = max(float(feature @ leverage_vector), 0.0)
        else:
            assert self.gram_inverse is not None
            leverage = max(float(feature @ self.gram_inverse @ feature), 0.0)
        variance = self.residual_variance * (1.0 + min(leverage, 25.0))
        return mean.reshape(self.output_shape), variance.reshape(self.output_shape)


@dataclass(frozen=True)
class GenerativeModel:
    graph_object: GraphObject
    trajectory_model: RidgeTrajectoryModel

    def predict(self, episode: Episode) -> PredictiveDistribution:
        displacement, variance = self.trajectory_model.predict(episode.descriptor)
        return PredictiveDistribution(
            method="generative_only",
            mean=self.graph_object.rest_positions[None, :, :] + displacement,
            variance=variance,
        )


@dataclass(frozen=True)
class PhysicsModel:
    graph_object: GraphObject
    posterior: ParameterPosterior
    config: CounterfactualBenchmarkConfig

    def predict_action(
        self,
        action: Action,
        condition: WorldCondition,
        *,
        method: str = "physics_only",
    ) -> PredictiveDistribution:
        trajectories = simulate_particles(
            self.graph_object,
            action,
            self.posterior.particles,
            condition.plan_model(),
            self.config.simulator,
        )
        mean = np.sum(
            self.posterior.weights[:, None, None, None] * trajectories,
            axis=0,
        )
        variance = np.sum(
            self.posterior.weights[:, None, None, None]
            * np.square(trajectories - mean[None, ...]),
            axis=0,
        )
        variance += self.config.predictive_variance_floor_m2
        return PredictiveDistribution(method=method, mean=mean, variance=variance)

    def predict(self, episode: Episode) -> PredictiveDistribution:
        return self.predict_action(episode.action, episode.condition)


@dataclass(frozen=True)
class HybridModel:
    physics: PhysicsModel
    residual_model: RidgeTrajectoryModel
    residual_scale: float
    validation_rmse_m: float

    def predict(self, episode: Episode) -> PredictiveDistribution:
        physics = self.physics.predict_action(
            episode.action,
            episode.condition,
            method="hybrid",
        )
        residual, residual_variance = self.residual_model.predict(episode.descriptor)
        return PredictiveDistribution(
            method="hybrid",
            mean=physics.mean + self.residual_scale * residual,
            variance=(physics.variance + self.residual_scale**2 * residual_variance),
        )


@dataclass(frozen=True)
class FittedBaselines:
    generative: GenerativeModel
    physics: PhysicsModel
    hybrid: HybridModel

    def predict_all(self, episode: Episode) -> tuple[PredictiveDistribution, ...]:
        return (
            self.generative.predict(episode),
            self.physics.predict(episode),
            self.hybrid.predict(episode),
        )


def fit_parameter_posterior(
    graph_object: GraphObject,
    training_episodes: tuple[Episode, ...],
    particles: np.ndarray,
    config: CounterfactualBenchmarkConfig,
) -> ParameterPosterior:
    """Fit a tempered grid posterior from sparse, temporally subsampled sensors."""

    if not training_episodes:
        raise ValueError("at least one training episode is required")
    particles = np.asarray(particles, dtype=float)
    log_likelihood = np.zeros(particles.shape[0], dtype=float)
    frames = np.arange(0, config.frame_count, config.fit_frame_stride)
    sensors = np.asarray(graph_object.sensor_nodes, dtype=int)
    variance = config.inference_noise_std_m**2

    for episode in training_episodes:
        trajectories = simulate_particles(
            graph_object,
            episode.action,
            particles,
            episode.condition.plan_model(),
            config.simulator,
        )
        residual = (
            trajectories[:, frames[:, None], sensors[None, :], :]
            - episode.observations[frames[:, None], sensors[None, :], :][None, ...]
        )
        squared_error = np.sum(np.square(residual), axis=(1, 2, 3))
        log_likelihood -= 0.5 * config.likelihood_power * squared_error / variance

    shifted = log_likelihood - float(np.max(log_likelihood))
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    return ParameterPosterior(
        particles=particles,
        weights=weights,
        log_likelihood=log_likelihood,
    )


def _fit_ridge_trajectory(
    descriptors: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    variance_floor: float,
) -> RidgeTrajectoryModel:
    descriptors = np.asarray(descriptors, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if descriptors.ndim != 2 or targets.ndim != 4:
        raise ValueError("descriptors and trajectory targets have invalid rank")
    if descriptors.shape[0] != targets.shape[0] or descriptors.shape[0] < 2:
        raise ValueError("at least two aligned training examples are required")
    if not np.all(np.isfinite(descriptors)) or not np.all(np.isfinite(targets)):
        raise ValueError("descriptors and trajectory targets must be finite")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")

    feature_mean = np.mean(descriptors, axis=0)
    feature_scale = np.std(descriptors, axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    standardized = (descriptors - feature_mean) / feature_scale
    design = np.column_stack([np.ones(descriptors.shape[0]), standardized])
    flat_targets = targets.reshape(targets.shape[0], -1)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = ridge * 1e-3
    gram = design.T @ design + penalty
    coefficients = np.linalg.solve(gram, design.T @ flat_targets)

    # Leave-one-out errors provide a non-degenerate uncertainty estimate even
    # when the feature dimension approaches the number of interactions.
    held_out_predictions = np.empty_like(flat_targets)
    for index in range(design.shape[0]):
        selected = np.arange(design.shape[0]) != index
        sub_gram = design[selected].T @ design[selected] + penalty
        sub_coefficients = np.linalg.solve(
            sub_gram,
            design[selected].T @ flat_targets[selected],
        )
        held_out_predictions[index] = design[index] @ sub_coefficients
    residual_variance = np.mean(np.square(flat_targets - held_out_predictions), axis=0)
    residual_variance += variance_floor
    return RidgeTrajectoryModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        gram_inverse=None,
        residual_variance=residual_variance,
        output_shape=tuple(targets.shape[1:]),
        gram_matrix=gram,
    )


def fit_baselines(
    training_episodes: tuple[Episode, ...],
    validation_episode: Episode,
    particles: np.ndarray,
    config: CounterfactualBenchmarkConfig,
) -> FittedBaselines:
    """Fit the three evidence-distinct baselines without touching test data."""

    if not training_episodes:
        raise ValueError("training episodes are required")
    graph_object = training_episodes[0].graph_object
    if any(
        episode.graph_object.name != graph_object.name for episode in training_episodes
    ):
        raise ValueError("all training episodes must belong to one object")

    descriptors = np.stack([episode.descriptor for episode in training_episodes])
    displacements = np.stack(
        [
            episode.observations - graph_object.rest_positions[None, :, :]
            for episode in training_episodes
        ]
    )
    generative = GenerativeModel(
        graph_object=graph_object,
        trajectory_model=_fit_ridge_trajectory(
            descriptors,
            displacements,
            ridge=config.generative_ridge,
            variance_floor=config.predictive_variance_floor_m2,
        ),
    )

    posterior = fit_parameter_posterior(
        graph_object,
        training_episodes,
        particles,
        config,
    )
    physics = PhysicsModel(
        graph_object=graph_object,
        posterior=posterior,
        config=config,
    )
    physics_training = [physics.predict(episode) for episode in training_episodes]
    residual_targets = np.stack(
        [
            episode.observations - prediction.mean
            for episode, prediction in zip(
                training_episodes, physics_training, strict=True
            )
        ]
    )
    residual_model = _fit_ridge_trajectory(
        descriptors,
        residual_targets,
        ridge=config.generative_ridge,
        variance_floor=config.predictive_variance_floor_m2,
    )

    validation_physics = physics.predict(validation_episode)
    validation_residual, _ = residual_model.predict(validation_episode.descriptor)
    scales = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0, 1.25])
    errors = [
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        validation_physics.mean
                        + scale * validation_residual
                        - validation_episode.observations
                    )
                )
            )
        )
        for scale in scales
    ]
    best_index = int(np.argmin(errors))
    hybrid = HybridModel(
        physics=physics,
        residual_model=residual_model,
        residual_scale=float(scales[best_index]),
        validation_rmse_m=float(errors[best_index]),
    )
    return FittedBaselines(
        generative=generative,
        physics=physics,
        hybrid=hybrid,
    )
