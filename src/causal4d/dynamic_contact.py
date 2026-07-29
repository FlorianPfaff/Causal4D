"""Dynamic contact-path inference for deformable-object interventions.

The existing Causal4D PhysTwin path represents one contact realization per
rollout. This module adds a backend-neutral finite-support approximation to a
path-valued contact posterior. Simulator backends remain responsible for
producing one trajectory for each candidate contact path; inference and
uncertainty propagation are handled here without reading held-out observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from statistics import NormalDist
from typing import Any

import numpy as np

from causal4d.weighting import log_weights_from_probabilities


class ContactRegime(IntEnum):
    """Discrete contact regime used by the dynamic intervention model."""

    INACTIVE = 0
    STICKING = 1
    SLIPPING = 2
    DETACHED = 3


CONTACT_REGIME_NAMES = tuple(regime.name.lower() for regime in ContactRegime)


def _readonly(values: np.ndarray, *, dtype: Any = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _normalized_weights(values: np.ndarray, *, name: str) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    return _readonly(weights / total)


def _probability(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class ContactTransitionConfig:
    """Action-conditioned transition probabilities for contact regimes."""

    activation_floor: float = 0.01
    activation_gain: float = 0.90
    slip_floor: float = 0.005
    slip_change_gain: float = 0.20
    release_floor: float = 0.005
    release_gain: float = 0.80
    slip_recovery_probability: float = 0.15
    reattachment_gain: float = 0.35
    maximum_paths: int = 64
    minimum_transition_probability: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "activation_floor",
            "activation_gain",
            "slip_floor",
            "slip_change_gain",
            "release_floor",
            "release_gain",
            "slip_recovery_probability",
            "reattachment_gain",
        ):
            _probability(getattr(self, name), name=name)
        if self.maximum_paths < 1:
            raise ValueError("maximum_paths must be positive")
        if not 0.0 <= self.minimum_transition_probability < 1.0:
            raise ValueError("minimum_transition_probability must lie in [0, 1)")


@dataclass(frozen=True)
class ContactPathPrior:
    """Finite beam approximation to an action-conditioned contact-path prior."""

    path_ids: tuple[str, ...]
    regime_paths: np.ndarray
    weights: np.ndarray
    retained_prior_mass: float

    def __post_init__(self) -> None:
        paths = _readonly(self.regime_paths, dtype=np.int8)
        weights = _normalized_weights(self.weights, name="contact-path weights")
        if paths.ndim != 2 or paths.shape[0] != len(weights):
            raise ValueError("regime_paths must have shape (K, T)")
        if paths.shape[1] < 1:
            raise ValueError("contact paths must contain at least one frame")
        if np.any(paths < 0) or np.any(paths >= len(CONTACT_REGIME_NAMES)):
            raise ValueError("contact paths contain an unknown regime")
        if len(self.path_ids) != len(weights) or len(set(self.path_ids)) != len(
            weights
        ):
            raise ValueError("path_ids must uniquely identify every contact path")
        if not np.isfinite(self.retained_prior_mass) or not (
            0.0 < self.retained_prior_mass <= 1.0 + 1e-12
        ):
            raise ValueError("retained_prior_mass must lie in (0, 1]")
        object.__setattr__(self, "regime_paths", paths)
        object.__setattr__(self, "weights", weights)


def contact_transition_matrix(
    activation: float,
    previous_activation: float,
    config: ContactTransitionConfig | None = None,
) -> np.ndarray:
    """Return one row-stochastic action-conditioned regime transition matrix."""

    settings = config or ContactTransitionConfig()
    current = _probability(activation, name="activation")
    previous = _probability(previous_activation, name="previous_activation")
    change = abs(current - previous)
    release_drive = max(1.0 - current, previous - current)

    matrix = np.zeros((len(ContactRegime), len(ContactRegime)), dtype=float)

    p_activate = np.clip(
        settings.activation_floor + settings.activation_gain * current,
        0.0,
        1.0,
    )
    matrix[ContactRegime.INACTIVE, ContactRegime.STICKING] = p_activate
    matrix[ContactRegime.INACTIVE, ContactRegime.INACTIVE] = 1.0 - p_activate

    p_detach = np.clip(
        settings.release_floor + settings.release_gain * release_drive,
        0.0,
        1.0,
    )
    p_slip = np.clip(
        settings.slip_floor + settings.slip_change_gain * change,
        0.0,
        1.0 - p_detach,
    )
    matrix[ContactRegime.STICKING, ContactRegime.DETACHED] = p_detach
    matrix[ContactRegime.STICKING, ContactRegime.SLIPPING] = p_slip
    matrix[ContactRegime.STICKING, ContactRegime.STICKING] = 1.0 - p_detach - p_slip

    p_recover = np.clip(
        settings.slip_recovery_probability * current,
        0.0,
        1.0 - p_detach,
    )
    matrix[ContactRegime.SLIPPING, ContactRegime.DETACHED] = p_detach
    matrix[ContactRegime.SLIPPING, ContactRegime.STICKING] = p_recover
    matrix[ContactRegime.SLIPPING, ContactRegime.SLIPPING] = 1.0 - p_detach - p_recover

    p_reattach = np.clip(settings.reattachment_gain * current, 0.0, 1.0)
    matrix[ContactRegime.DETACHED, ContactRegime.STICKING] = p_reattach
    matrix[ContactRegime.DETACHED, ContactRegime.DETACHED] = 1.0 - p_reattach

    if not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("contact transition matrix is not row stochastic")
    return matrix


def _path_id(path: tuple[int, ...]) -> str:
    runs: list[str] = []
    start = 0
    for stop in range(1, len(path) + 1):
        if stop == len(path) or path[stop] != path[start]:
            name = CONTACT_REGIME_NAMES[path[start]]
            runs.append(f"{name}:{start}-{stop - 1}")
            start = stop
    return "|".join(runs)


def enumerate_contact_paths(
    command_activation: np.ndarray,
    *,
    config: ContactTransitionConfig | None = None,
    initial_probabilities: np.ndarray | None = None,
) -> ContactPathPrior:
    """Enumerate the highest-probability contact paths with deterministic pruning."""

    settings = config or ContactTransitionConfig()
    activation = np.asarray(command_activation, dtype=float)
    if activation.ndim != 1 or len(activation) < 1:
        raise ValueError("command_activation must be a nonempty vector")
    if not np.all(np.isfinite(activation)) or np.any(
        (activation < 0.0) | (activation > 1.0)
    ):
        raise ValueError("command_activation must lie in [0, 1]")
    if initial_probabilities is None:
        initial = np.zeros(len(ContactRegime), dtype=float)
        initial[ContactRegime.INACTIVE] = 1.0
    else:
        initial = _normalized_weights(
            initial_probabilities,
            name="initial regime probabilities",
        )
        if initial.shape != (len(ContactRegime),):
            raise ValueError("initial regime probabilities must identify four regimes")

    beam: list[tuple[tuple[int, ...], float]] = [
        ((int(regime),), float(np.log(probability)))
        for regime, probability in enumerate(initial)
        if probability > settings.minimum_transition_probability
    ]
    retained_mass = 1.0
    for frame in range(1, len(activation)):
        transition = contact_transition_matrix(
            activation[frame],
            activation[frame - 1],
            settings,
        )
        expanded: list[tuple[tuple[int, ...], float]] = []
        for path, log_probability in beam:
            for next_regime, probability in enumerate(transition[path[-1]]):
                if probability <= settings.minimum_transition_probability:
                    continue
                expanded.append(
                    (
                        (*path, next_regime),
                        log_probability + float(np.log(probability)),
                    )
                )
        if not expanded:
            raise RuntimeError("contact-path beam lost all probability mass")
        expanded.sort(key=lambda item: (-item[1], item[0]))
        beam = expanded[: settings.maximum_paths]
        maximum = max(log_probability for _, log_probability in beam)
        retained_mass = float(
            np.exp(maximum)
            * np.sum([np.exp(log_probability - maximum) for _, log_probability in beam])
        )

    log_weights = np.asarray([value for _, value in beam], dtype=float)
    maximum = float(np.max(log_weights))
    weights = np.exp(log_weights - maximum)
    weights /= np.sum(weights)
    paths = np.asarray([path for path, _ in beam], dtype=np.int8)
    return ContactPathPrior(
        path_ids=tuple(_path_id(path) for path, _ in beam),
        regime_paths=paths,
        weights=weights,
        retained_prior_mass=min(retained_mass, 1.0),
    )


@dataclass(frozen=True)
class DynamicContactPathBank:
    """Precomputed simulator trajectories indexed by complete contact paths."""

    path_ids: tuple[str, ...]
    regime_paths: np.ndarray
    trajectories_m: np.ndarray
    prior_weights: np.ndarray
    base_variance_m2: np.ndarray | float

    def __post_init__(self) -> None:
        trajectories = _readonly(self.trajectories_m, dtype=float)
        paths = _readonly(self.regime_paths, dtype=np.int8)
        weights = _normalized_weights(self.prior_weights, name="path-bank weights")
        if trajectories.ndim != 4 or trajectories.shape[-1] not in {2, 3}:
            raise ValueError("trajectories_m must have shape (K, T, N, 2|3)")
        path_count, frame_count, node_count, coordinate_count = trajectories.shape
        if paths.shape != (path_count, frame_count):
            raise ValueError("regime_paths must match path and frame counts")
        if len(weights) != path_count:
            raise ValueError("prior_weights must identify every path")
        if len(self.path_ids) != path_count or len(set(self.path_ids)) != path_count:
            raise ValueError("path_ids must uniquely identify every path")
        if np.any(paths < 0) or np.any(paths >= len(CONTACT_REGIME_NAMES)):
            raise ValueError("regime_paths contain an unknown regime")
        if not np.all(np.isfinite(trajectories)):
            raise ValueError("path trajectories must be finite")

        variance = np.asarray(self.base_variance_m2, dtype=float)
        allowed_shapes = {
            (),
            (node_count, coordinate_count),
            (path_count, node_count, coordinate_count),
            (path_count, frame_count, node_count, coordinate_count),
        }
        if variance.shape not in allowed_shapes:
            raise ValueError(
                "base_variance_m2 must be scalar or have shape (N, C), "
                "(K, N, C), or (K, T, N, C)"
            )
        if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("base_variance_m2 must be finite and nonnegative")
        variance = _readonly(variance, dtype=float)

        object.__setattr__(self, "trajectories_m", trajectories)
        object.__setattr__(self, "regime_paths", paths)
        object.__setattr__(self, "prior_weights", weights)
        object.__setattr__(self, "base_variance_m2", variance)


@dataclass(frozen=True)
class DynamicContactInferenceConfig:
    """Robust prefix likelihood and intervention-conditioned variance settings."""

    observation_scale_m: float = 0.01
    degrees_of_freedom: float = 4.0
    likelihood_power: float = 1.0
    dynamic_likelihood_weight: float = 0.25
    switch_variance_m2: float = 4e-6
    command_change_variance_m2: float = 1e-6
    ood_variance_m2: float = 4e-6
    confidence_level: float = 0.90

    def __post_init__(self) -> None:
        positive = (
            self.observation_scale_m,
            self.degrees_of_freedom,
            self.likelihood_power,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "likelihood scale, degrees of freedom, and power must be positive"
            )
        nonnegative = (
            self.dynamic_likelihood_weight,
            self.switch_variance_m2,
            self.command_change_variance_m2,
            self.ood_variance_m2,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("likelihood and variance weights must be nonnegative")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")


@dataclass(frozen=True)
class DynamicContactPosterior:
    """Posterior moments and contact-regime probabilities over a path bank."""

    path_ids: tuple[str, ...]
    weights: np.ndarray
    mean_m: np.ndarray
    variance_m2: np.ndarray
    interval_lower_m: np.ndarray
    interval_upper_m: np.ndarray
    conditional_variance_m2: np.ndarray
    regime_probabilities: np.ndarray
    switch_probability: np.ndarray
    evidence_frame_stop: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        weights = _normalized_weights(self.weights, name="dynamic posterior weights")
        mean = _readonly(self.mean_m, dtype=float)
        variance = _readonly(self.variance_m2, dtype=float)
        lower = _readonly(self.interval_lower_m, dtype=float)
        upper = _readonly(self.interval_upper_m, dtype=float)
        conditional = _readonly(self.conditional_variance_m2, dtype=float)
        regime = _readonly(self.regime_probabilities, dtype=float)
        switch = _readonly(self.switch_probability, dtype=float)
        if len(self.path_ids) != len(weights):
            raise ValueError("path_ids must match posterior weights")
        if mean.ndim != 3 or mean.shape[-1] not in {2, 3}:
            raise ValueError("posterior mean must have shape (T, N, 2|3)")
        if (
            variance.shape != mean.shape
            or lower.shape != mean.shape
            or upper.shape != mean.shape
        ):
            raise ValueError("posterior variance and intervals must match the mean")
        if conditional.shape != (len(weights), *mean.shape):
            raise ValueError("conditional variance must have shape (K, T, N, C)")
        if regime.shape != (mean.shape[0], len(ContactRegime)):
            raise ValueError("regime probabilities must have shape (T, 4)")
        if switch.shape != (mean.shape[0],):
            raise ValueError("switch_probability must have shape (T,)")
        if not 1 <= self.evidence_frame_stop < mean.shape[0]:
            raise ValueError("evidence_frame_stop must leave held-out frames")
        if np.any(variance <= 0.0) or np.any(conditional < 0.0):
            raise ValueError("posterior variances must be positive")
        if np.any(lower > upper):
            raise ValueError("posterior interval lower bound exceeds upper bound")
        if not np.allclose(np.sum(regime, axis=1), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("regime probabilities must sum to one per frame")
        if np.any((switch < 0.0) | (switch > 1.0)):
            raise ValueError("switch probabilities must lie in [0, 1]")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "interval_lower_m", lower)
        object.__setattr__(self, "interval_upper_m", upper)
        object.__setattr__(self, "conditional_variance_m2", conditional)
        object.__setattr__(self, "regime_probabilities", regime)
        object.__setattr__(self, "switch_probability", switch)

    @property
    def map_path_id(self) -> str:
        """Identifier of the maximum-posterior contact path."""

        return self.path_ids[int(np.argmax(self.weights))]


def _base_variance_schedule(bank: DynamicContactPathBank) -> np.ndarray:
    path_count, frame_count, node_count, coordinate_count = bank.trajectories_m.shape
    variance = np.asarray(bank.base_variance_m2, dtype=float)
    if variance.ndim == 0:
        return np.full(bank.trajectories_m.shape, float(variance), dtype=float)
    if variance.shape == (node_count, coordinate_count):
        return np.broadcast_to(
            variance[None, None],
            bank.trajectories_m.shape,
        ).copy()
    if variance.shape == (path_count, node_count, coordinate_count):
        return np.broadcast_to(
            variance[:, None],
            bank.trajectories_m.shape,
        ).copy()
    if variance.shape == (
        path_count,
        frame_count,
        node_count,
        coordinate_count,
    ):
        return variance.copy()
    raise RuntimeError("validated variance shape became unsupported")


def contact_conditioned_variance(
    bank: DynamicContactPathBank,
    command_activation: np.ndarray,
    *,
    ood_distance: np.ndarray | None = None,
    config: DynamicContactInferenceConfig | None = None,
) -> np.ndarray:
    """Forecast conditional variance with switch, command, and OOD increments."""

    settings = config or DynamicContactInferenceConfig()
    activation = np.asarray(command_activation, dtype=float)
    frame_count = bank.trajectories_m.shape[1]
    if activation.shape != (frame_count,):
        raise ValueError("command_activation must match the rollout frame count")
    if not np.all(np.isfinite(activation)) or np.any(
        (activation < 0.0) | (activation > 1.0)
    ):
        raise ValueError("command_activation must lie in [0, 1]")
    if ood_distance is None:
        ood = np.zeros(frame_count, dtype=float)
    else:
        ood = np.asarray(ood_distance, dtype=float)
        if (
            ood.shape != (frame_count,)
            or not np.all(np.isfinite(ood))
            or np.any(ood < 0.0)
        ):
            raise ValueError("ood_distance must be a nonnegative frame vector")

    switches = np.zeros_like(bank.regime_paths, dtype=float)
    switches[:, 1:] = bank.regime_paths[:, 1:] != bank.regime_paths[:, :-1]
    cumulative_switches = np.cumsum(switches, axis=1)
    command_change = np.zeros(frame_count, dtype=float)
    command_change[1:] = np.diff(activation)
    cumulative_command_energy = np.cumsum(np.square(command_change))
    cumulative_ood_energy = np.cumsum(np.square(ood))
    increment = (
        settings.switch_variance_m2 * cumulative_switches
        + settings.command_change_variance_m2 * cumulative_command_energy[None]
        + settings.ood_variance_m2 * cumulative_ood_energy[None]
    )
    return _base_variance_schedule(bank) + increment[:, :, None, None]


def _coordinate_mask(observations: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    finite = np.isfinite(observations)
    if mask is None:
        return finite
    supplied = np.asarray(mask, dtype=bool)
    if supplied.shape == observations.shape[:2]:
        supplied = np.repeat(supplied[:, :, None], observations.shape[2], axis=2)
    if supplied.shape != observations.shape:
        raise ValueError("mask must have shape (T, N) or (T, N, C)")
    return finite & supplied


def _student_t_mean_log_score(
    residual: np.ndarray,
    valid: np.ndarray,
    scale_m: np.ndarray,
    degrees_of_freedom: float,
) -> np.ndarray:
    standardized = residual / scale_m
    terms = (
        -0.5
        * (degrees_of_freedom + 1.0)
        * np.log1p(np.square(standardized) / degrees_of_freedom)
    )
    valid_float = np.asarray(valid, dtype=float)[None]
    count = np.sum(valid_float, axis=(1, 2, 3))
    if np.any(count <= 0.0):
        raise ValueError("dynamic contact update has no valid coordinates")
    return (
        np.sum(
            np.where(valid_float > 0.0, terms, 0.0),
            axis=(1, 2, 3),
        )
        / count
    )


def _normalized_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("dynamic contact posterior normalization failed")
    return weights / total


def infer_dynamic_contact_posterior(
    bank: DynamicContactPathBank,
    observations_m: np.ndarray,
    *,
    prefix_frame_count: int,
    command_activation: np.ndarray,
    mask: np.ndarray | None = None,
    ood_distance: np.ndarray | None = None,
    config: DynamicContactInferenceConfig | None = None,
) -> DynamicContactPosterior:
    """Infer a contact-path posterior from a prefix and predict the held-out suffix."""

    settings = config or DynamicContactInferenceConfig()
    observations = np.asarray(observations_m, dtype=float)
    expected_shape = bank.trajectories_m.shape[1:]
    if observations.shape != expected_shape:
        raise ValueError(f"observations_m must have shape {expected_shape}")
    if not 1 <= prefix_frame_count < expected_shape[0]:
        raise ValueError("prefix_frame_count must be nonempty and leave a future")
    valid = _coordinate_mask(observations, mask)
    prefix_valid = valid[:prefix_frame_count]
    if not np.any(prefix_valid):
        raise ValueError("contact update prefix contains no valid observations")

    conditional_variance = contact_conditioned_variance(
        bank,
        command_activation,
        ood_distance=ood_distance,
        config=settings,
    )
    position_scale = np.sqrt(
        settings.observation_scale_m**2 + conditional_variance[:, :prefix_frame_count]
    )
    position_score = _student_t_mean_log_score(
        bank.trajectories_m[:, :prefix_frame_count]
        - observations[None, :prefix_frame_count],
        prefix_valid,
        position_scale,
        settings.degrees_of_freedom,
    )
    score = position_score
    if settings.dynamic_likelihood_weight > 0.0 and prefix_frame_count >= 2:
        predicted_delta = np.diff(
            bank.trajectories_m[:, :prefix_frame_count],
            axis=1,
        )
        observed_delta = np.diff(observations[:prefix_frame_count], axis=0)
        delta_valid = prefix_valid[1:] & prefix_valid[:-1]
        delta_variance = (
            2.0 * settings.observation_scale_m**2
            + conditional_variance[:, 1:prefix_frame_count]
            + conditional_variance[:, : prefix_frame_count - 1]
        )
        if np.any(delta_valid):
            dynamic_score = _student_t_mean_log_score(
                predicted_delta - observed_delta[None],
                delta_valid,
                np.sqrt(delta_variance),
                settings.degrees_of_freedom,
            )
            score = score + settings.dynamic_likelihood_weight * dynamic_score

    log_prior = log_weights_from_probabilities(
        bank.prior_weights,
        name="dynamic contact prior",
    )
    weights = _normalized_log_weights(log_prior + settings.likelihood_power * score)
    components = bank.trajectories_m
    mean = np.einsum("k,ktnc->tnc", weights, components)
    centered = components - mean[None]
    epistemic = np.einsum("k,ktnc->tnc", weights, np.square(centered))
    conditional = np.einsum("k,ktnc->tnc", weights, conditional_variance)
    variance = np.maximum(epistemic + conditional, np.finfo(float).tiny)
    z_score = NormalDist().inv_cdf(0.5 * (1.0 + settings.confidence_level))
    margin = z_score * np.sqrt(variance)

    regime_probabilities = np.zeros(
        (expected_shape[0], len(ContactRegime)),
        dtype=float,
    )
    for regime in ContactRegime:
        regime_probabilities[:, regime] = np.einsum(
            "k,kt->t",
            weights,
            bank.regime_paths == regime,
        )
    switch_probability = np.zeros(expected_shape[0], dtype=float)
    switch_probability[1:] = np.einsum(
        "k,kt->t",
        weights,
        bank.regime_paths[:, 1:] != bank.regime_paths[:, :-1],
    )

    return DynamicContactPosterior(
        path_ids=bank.path_ids,
        weights=weights,
        mean_m=mean,
        variance_m2=variance,
        interval_lower_m=mean - margin,
        interval_upper_m=mean + margin,
        conditional_variance_m2=conditional_variance,
        regime_probabilities=regime_probabilities,
        switch_probability=switch_probability,
        evidence_frame_stop=prefix_frame_count,
        metadata={
            "model": "dynamic_contact_path_bank",
            "contact_regimes": list(CONTACT_REGIME_NAMES),
            "prefix_frame_count": prefix_frame_count,
            "future_observations_read": 0,
            "config": asdict(settings),
        },
    )


def first_activation_frame(regime_path: np.ndarray) -> int | None:
    """Return the first sticking/slipping frame of one contact path."""

    path = np.asarray(regime_path, dtype=int)
    active = np.flatnonzero(
        (path == ContactRegime.STICKING) | (path == ContactRegime.SLIPPING)
    )
    return int(active[0]) if len(active) else None
