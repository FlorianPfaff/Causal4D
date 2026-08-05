"""Prefix-only Bayesian inference over joint multi-contact trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from causal4d._multi_contact_common import (
    MULTI_CONTACT_SCHEDULE_SCHEMA_VERSION,
    activation_matrix,
    integer_array,
    normalized_weights,
    probability_mass,
    real_array,
    readonly,
    schedule_identity,
    validate_identifiers,
)
from causal4d._multi_contact_prior import MultiContactPathPrior
from causal4d.dynamic_contact import (
    CONTACT_REGIME_NAMES,
    ContactRegime,
    DynamicContactInferenceConfig,
)
from causal4d.weighting import log_weights_from_probabilities


@dataclass(frozen=True)
class MultiContactPathBank:
    """Continuously simulated trajectories indexed by complete joint schedules."""

    contact_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    regime_paths: np.ndarray
    trajectories_m: np.ndarray
    prior_weights: np.ndarray
    base_variance_m2: np.ndarray | float
    retained_prior_mass: float = 1.0

    def __post_init__(self) -> None:
        trajectories = readonly(
            real_array(self.trajectories_m, name="trajectories_m")
        )
        paths = integer_array(
            self.regime_paths,
            name="regime_paths",
            dtype=np.int8,
        )
        weights = normalized_weights(
            self.prior_weights,
            name="multi-contact bank weights",
        )
        if trajectories.ndim != 4 or trajectories.shape[-1] not in {2, 3}:
            raise ValueError("trajectories_m must have shape (K, T, N, 2|3)")
        path_count, frame_count, node_count, coordinate_count = trajectories.shape
        if (
            paths.ndim != 3
            or paths.shape[0] != path_count
            or paths.shape[2] != frame_count
        ):
            raise ValueError("regime_paths must have shape (K, G, T)")
        contact_count = paths.shape[1]
        if contact_count < 1:
            raise ValueError("regime_paths must identify at least one contact")
        contact_ids = validate_identifiers(
            self.contact_ids,
            expected_count=contact_count,
            name="contact_ids",
        )
        path_ids = validate_identifiers(
            self.path_ids,
            expected_count=path_count,
            name="path_ids",
        )
        if len(weights) != path_count:
            raise ValueError("prior_weights must identify every path")
        if np.any(paths < 0) or np.any(paths >= len(CONTACT_REGIME_NAMES)):
            raise ValueError("regime_paths contain an unknown contact regime")
        variance = real_array(self.base_variance_m2, name="base_variance_m2")
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
        if np.any(variance < 0.0):
            raise ValueError("base_variance_m2 must be nonnegative")
        retained = probability_mass(
            self.retained_prior_mass,
            name="retained_prior_mass",
        )
        object.__setattr__(self, "contact_ids", contact_ids)
        object.__setattr__(self, "path_ids", path_ids)
        object.__setattr__(self, "regime_paths", paths)
        object.__setattr__(self, "trajectories_m", trajectories)
        object.__setattr__(self, "prior_weights", weights)
        object.__setattr__(self, "base_variance_m2", readonly(variance))
        object.__setattr__(self, "retained_prior_mass", retained)

    @property
    def schedule_identity(self) -> str:
        """Content identity shared with the prior used to build the bank."""

        return schedule_identity(
            self.contact_ids,
            self.path_ids,
            self.regime_paths,
            self.prior_weights,
            self.retained_prior_mass,
        )

    @classmethod
    def from_prior(
        cls,
        prior: MultiContactPathPrior,
        trajectories_m: np.ndarray,
        *,
        base_variance_m2: np.ndarray | float,
    ) -> MultiContactPathBank:
        """Attach continuous simulator trajectories to an enumerated joint prior."""

        if not isinstance(prior, MultiContactPathPrior):
            raise ValueError("prior must be a MultiContactPathPrior")
        return cls(
            contact_ids=prior.contact_ids,
            path_ids=prior.path_ids,
            regime_paths=prior.regime_paths,
            trajectories_m=trajectories_m,
            prior_weights=prior.weights,
            base_variance_m2=base_variance_m2,
            retained_prior_mass=prior.retained_prior_mass,
        )


@dataclass(frozen=True)
class MultiContactPosterior:
    """Predictive moments and per-contact regime marginals over a joint bank."""

    contact_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    weights: np.ndarray
    mean_m: np.ndarray
    variance_m2: np.ndarray
    interval_lower_m: np.ndarray
    interval_upper_m: np.ndarray
    conditional_variance_m2: np.ndarray
    regime_probabilities: np.ndarray
    switch_probability: np.ndarray
    any_switch_probability: np.ndarray
    evidence_frame_stop: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        weights = normalized_weights(
            self.weights,
            name="multi-contact posterior weights",
        )
        mean = readonly(real_array(self.mean_m, name="mean_m"))
        variance = readonly(real_array(self.variance_m2, name="variance_m2"))
        lower = readonly(real_array(self.interval_lower_m, name="interval_lower_m"))
        upper = readonly(real_array(self.interval_upper_m, name="interval_upper_m"))
        conditional = readonly(
            real_array(
                self.conditional_variance_m2,
                name="conditional_variance_m2",
            )
        )
        regime = readonly(
            real_array(self.regime_probabilities, name="regime_probabilities")
        )
        switch = readonly(
            real_array(self.switch_probability, name="switch_probability")
        )
        any_switch = readonly(
            real_array(
                self.any_switch_probability,
                name="any_switch_probability",
            )
        )
        contact_ids = validate_identifiers(
            self.contact_ids,
            expected_count=len(self.contact_ids),
            name="contact_ids",
        )
        path_ids = validate_identifiers(
            self.path_ids,
            expected_count=len(weights),
            name="path_ids",
        )
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
        if regime.shape != (len(contact_ids), mean.shape[0], len(ContactRegime)):
            raise ValueError("regime probabilities must have shape (G, T, 4)")
        if switch.shape != (len(contact_ids), mean.shape[0]):
            raise ValueError("switch_probability must have shape (G, T)")
        if any_switch.shape != (mean.shape[0],):
            raise ValueError("any_switch_probability must have shape (T,)")
        if (
            type(self.evidence_frame_stop) is not int
            or not 1 <= self.evidence_frame_stop < mean.shape[0]
        ):
            raise ValueError(
                "evidence_frame_stop must be an integer leaving held-out frames"
            )
        if np.any(variance <= 0.0) or np.any(conditional < 0.0):
            raise ValueError("posterior variances must be positive")
        if np.any(lower > upper):
            raise ValueError("posterior interval lower bound exceeds upper bound")
        if not np.allclose(np.sum(regime, axis=2), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError(
                "regime probabilities must sum to one per contact and frame"
            )
        if np.any((switch < 0.0) | (switch > 1.0)) or np.any(
            (any_switch < 0.0) | (any_switch > 1.0)
        ):
            raise ValueError("switch probabilities must lie in [0, 1]")
        object.__setattr__(self, "contact_ids", contact_ids)
        object.__setattr__(self, "path_ids", path_ids)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "interval_lower_m", lower)
        object.__setattr__(self, "interval_upper_m", upper)
        object.__setattr__(self, "conditional_variance_m2", conditional)
        object.__setattr__(self, "regime_probabilities", regime)
        object.__setattr__(self, "switch_probability", switch)
        object.__setattr__(self, "any_switch_probability", any_switch)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def map_path_id(self) -> str:
        """Identifier of the maximum-posterior joint contact path."""

        return self.path_ids[int(np.argmax(self.weights))]

    @property
    def active_probability(self) -> np.ndarray:
        """Per-contact probability of sticking or slipping at every frame."""

        result = (
            self.regime_probabilities[..., ContactRegime.STICKING]
            + self.regime_probabilities[..., ContactRegime.SLIPPING]
        )
        return readonly(result)


def _base_variance_schedule(bank: MultiContactPathBank) -> np.ndarray:
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


def multi_contact_conditioned_variance(
    bank: MultiContactPathBank,
    command_activation: np.ndarray,
    *,
    ood_distance: np.ndarray | None = None,
    config: DynamicContactInferenceConfig | None = None,
) -> np.ndarray:
    """Forecast variance from all contact switches, commands, and OOD distances."""

    settings = config or DynamicContactInferenceConfig()
    contact_count = bank.regime_paths.shape[1]
    frame_count = bank.trajectories_m.shape[1]
    activation = activation_matrix(
        command_activation,
        contact_count=contact_count,
        frame_count=frame_count,
    )
    if ood_distance is None:
        ood_frame_energy = np.zeros(frame_count, dtype=float)
    else:
        supplied = real_array(ood_distance, name="ood_distance")
        if supplied.shape == (frame_count,):
            if np.any(supplied < 0.0):
                raise ValueError("ood_distance must be nonnegative")
            ood_frame_energy = np.square(supplied)
        elif supplied.shape == (contact_count, frame_count):
            if np.any(supplied < 0.0):
                raise ValueError("ood_distance must be nonnegative")
            ood_frame_energy = np.sum(np.square(supplied), axis=0)
        else:
            raise ValueError("ood_distance must have shape (T,) or (G, T)")

    switches = np.zeros_like(bank.regime_paths, dtype=float)
    switches[:, :, 1:] = bank.regime_paths[:, :, 1:] != bank.regime_paths[:, :, :-1]
    cumulative_switches = np.cumsum(np.sum(switches, axis=1), axis=1)
    command_change = np.zeros_like(activation)
    command_change[:, 1:] = np.diff(activation, axis=1)
    cumulative_command_energy = np.cumsum(
        np.sum(np.square(command_change), axis=0)
    )
    cumulative_ood_energy = np.cumsum(ood_frame_energy)
    increment = (
        settings.switch_variance_m2 * cumulative_switches
        + settings.command_change_variance_m2 * cumulative_command_energy[None]
        + settings.ood_variance_m2 * cumulative_ood_energy[None]
    )
    return _base_variance_schedule(bank) + increment[:, :, None, None]


def _prefix_coordinate_mask(
    observations: np.ndarray,
    mask: np.ndarray | None,
    *,
    full_shape: tuple[int, int, int],
    prefix_frame_count: int,
) -> np.ndarray:
    finite = np.isfinite(observations)
    if mask is None:
        return finite
    supplied = np.asarray(mask)
    if supplied.dtype.kind != "b":
        raise ValueError("mask must contain booleans")
    if supplied.shape == full_shape[:2]:
        prefix = supplied[:prefix_frame_count, :, None]
        prefix = np.repeat(prefix, full_shape[2], axis=2)
    elif supplied.shape == full_shape:
        prefix = supplied[:prefix_frame_count]
    else:
        raise ValueError("mask must have shape (T, N) or (T, N, C)")
    return finite & prefix


def _student_t_mean_log_score(
    residual: np.ndarray,
    valid: np.ndarray,
    scale_m: np.ndarray,
    degrees_of_freedom: float,
) -> np.ndarray:
    standardized = residual / scale_m
    terms = -0.5 * (degrees_of_freedom + 1.0) * np.log1p(
        np.square(standardized) / degrees_of_freedom
    )
    valid_float = np.asarray(valid, dtype=float)[None]
    count = np.sum(valid_float, axis=(1, 2, 3))
    if np.any(count <= 0.0):
        raise ValueError("multi-contact update has no valid coordinates")
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
        raise RuntimeError("multi-contact posterior normalization failed")
    return weights / total


def infer_multi_contact_posterior(
    bank: MultiContactPathBank,
    observations_m: np.ndarray,
    *,
    prefix_frame_count: int,
    command_activation: np.ndarray,
    mask: np.ndarray | None = None,
    ood_distance: np.ndarray | None = None,
    config: DynamicContactInferenceConfig | None = None,
) -> MultiContactPosterior:
    """Infer a joint contact-path posterior from an observation prefix only."""

    settings = config or DynamicContactInferenceConfig()
    observations = real_array(
        observations_m,
        name="observations_m",
        require_finite=False,
    )
    expected_shape = bank.trajectories_m.shape[1:]
    if observations.shape != expected_shape:
        raise ValueError(f"observations_m must have shape {expected_shape}")
    if type(prefix_frame_count) is not int or not (
        1 <= prefix_frame_count < expected_shape[0]
    ):
        raise ValueError(
            "prefix_frame_count must be an integer leaving held-out frames"
        )
    prefix_observations = observations[:prefix_frame_count]
    prefix_valid = _prefix_coordinate_mask(
        prefix_observations,
        mask,
        full_shape=expected_shape,
        prefix_frame_count=prefix_frame_count,
    )
    if not np.any(prefix_valid):
        raise ValueError("multi-contact update prefix contains no valid observations")

    conditional_variance = multi_contact_conditioned_variance(
        bank,
        command_activation,
        ood_distance=ood_distance,
        config=settings,
    )
    position_scale = np.sqrt(
        settings.observation_scale_m**2
        + conditional_variance[:, :prefix_frame_count]
    )
    position_score = _student_t_mean_log_score(
        bank.trajectories_m[:, :prefix_frame_count]
        - prefix_observations[None],
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
        observed_delta = np.diff(prefix_observations, axis=0)
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
        name="multi-contact prior",
    )
    weights = _normalized_log_weights(
        log_prior + settings.likelihood_power * score
    )
    components = bank.trajectories_m
    mean = np.einsum("k,ktnc->tnc", weights, components)
    centered = components - mean[None]
    epistemic = np.einsum("k,ktnc->tnc", weights, np.square(centered))
    conditional = np.einsum("k,ktnc->tnc", weights, conditional_variance)
    variance = np.maximum(epistemic + conditional, np.finfo(float).tiny)
    z_score = NormalDist().inv_cdf(0.5 * (1.0 + settings.confidence_level))
    margin = z_score * np.sqrt(variance)

    contact_count = bank.regime_paths.shape[1]
    regime_probabilities = np.zeros(
        (contact_count, expected_shape[0], len(ContactRegime)),
        dtype=float,
    )
    for regime in ContactRegime:
        regime_probabilities[:, :, regime] = np.einsum(
            "k,kgt->gt",
            weights,
            bank.regime_paths == regime,
        )
    switches = np.zeros_like(bank.regime_paths, dtype=bool)
    switches[:, :, 1:] = (
        bank.regime_paths[:, :, 1:] != bank.regime_paths[:, :, :-1]
    )
    switch_probability = np.einsum("k,kgt->gt", weights, switches)
    any_switch_probability = np.einsum(
        "k,kt->t",
        weights,
        np.any(switches, axis=1),
    )
    return MultiContactPosterior(
        contact_ids=bank.contact_ids,
        path_ids=bank.path_ids,
        weights=weights,
        mean_m=mean,
        variance_m2=variance,
        interval_lower_m=mean - margin,
        interval_upper_m=mean + margin,
        conditional_variance_m2=conditional_variance,
        regime_probabilities=regime_probabilities,
        switch_probability=switch_probability,
        any_switch_probability=any_switch_probability,
        evidence_frame_stop=prefix_frame_count,
        metadata={
            "model": "factorized_multi_contact_path_bank",
            "contact_ids": list(bank.contact_ids),
            "contact_regimes": list(CONTACT_REGIME_NAMES),
            "prefix_frame_count": prefix_frame_count,
            "future_observations_read": 0,
            "retained_prior_mass": bank.retained_prior_mass,
            "schedule_identity": bank.schedule_identity,
            "schedule_schema_version": MULTI_CONTACT_SCHEDULE_SCHEMA_VERSION,
            "config": asdict(settings),
        },
    )
