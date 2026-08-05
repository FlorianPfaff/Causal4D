"""Prefix-only Bayesian inference over joint multi-contact trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Real
from typing import Any

import numpy as np
from scipy.special import ndtr

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
from causal4d._student_t_likelihood import student_t_mean_log_score
from causal4d.dynamic_contact import (
    CONTACT_REGIME_NAMES,
    ContactRegime,
    DynamicContactInferenceConfig,
)
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.weighting import log_weights_from_probabilities


MULTI_CONTACT_ROLLOUT_SCHEMA_VERSION = "causal4d.multi_contact_rollout.v1"


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string when supplied")
    return value


def _array_record(values: np.ndarray, *, dtype: str) -> dict[str, Any]:
    canonical = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _rollout_identity(
    *,
    schedule_id: str,
    trajectories_m: np.ndarray,
    base_variance_m2: np.ndarray,
    replay_result_identity: str | None,
    frame_times_s: np.ndarray | None,
) -> str:
    payload = {
        "schema_version": MULTI_CONTACT_ROLLOUT_SCHEMA_VERSION,
        "schedule_identity": schedule_id,
        "replay_result_identity": replay_result_identity,
        "trajectories_m": _array_record(trajectories_m, dtype="<f8"),
        "base_variance_m2": _array_record(base_variance_m2, dtype="<f8"),
        "frame_times_s": (
            None if frame_times_s is None else _array_record(frame_times_s, dtype="<f8")
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    replay_result_identity: str | None = None
    frame_times_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        trajectories = readonly(real_array(self.trajectories_m, name="trajectories_m"))
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
        replay_identity: str | None = _optional_identity(
            self.replay_result_identity,
            name="replay_result_identity",
        )
        frame_times: np.ndarray | None
        if self.frame_times_s is None:
            frame_times = None
        else:
            frame_times = readonly(real_array(self.frame_times_s, name="frame_times_s"))
            if frame_times.shape != (frame_count,):
                raise ValueError("frame_times_s must match the rollout frame count")
            if frame_count > 1 and np.any(np.diff(frame_times) <= 0.0):
                raise ValueError("frame_times_s must be strictly increasing")

        object.__setattr__(self, "contact_ids", contact_ids)
        object.__setattr__(self, "path_ids", path_ids)
        object.__setattr__(self, "regime_paths", paths)
        object.__setattr__(self, "trajectories_m", trajectories)
        object.__setattr__(self, "prior_weights", weights)
        object.__setattr__(self, "base_variance_m2", readonly(variance))
        object.__setattr__(self, "retained_prior_mass", retained)
        object.__setattr__(self, "replay_result_identity", replay_identity)
        object.__setattr__(self, "frame_times_s", frame_times)

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

    @property
    def rollout_identity(self) -> str:
        """Bind schedule, replay result, timebase, trajectories, and variance."""

        return _rollout_identity(
            schedule_id=self.schedule_identity,
            trajectories_m=self.trajectories_m,
            base_variance_m2=np.asarray(self.base_variance_m2),
            replay_result_identity=self.replay_result_identity,
            frame_times_s=self.frame_times_s,
        )

    @property
    def replay_bound(self) -> bool:
        """Whether the bank binds an external replay result and explicit timebase."""

        return (
            self.replay_result_identity is not None and self.frame_times_s is not None
        )

    @classmethod
    def from_prior(
        cls,
        prior: MultiContactPathPrior,
        trajectories_m: np.ndarray,
        *,
        base_variance_m2: np.ndarray | float,
        replay_result_identity: str | None = None,
        frame_times_s: np.ndarray | None = None,
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
            replay_result_identity=replay_result_identity,
            frame_times_s=frame_times_s,
        )


@dataclass(frozen=True)
class MultiContactInferencePolicy:
    """Fail-closed support and replay-binding requirements."""

    minimum_retained_prior_mass: float = 0.0
    maximum_omitted_posterior_mass: float = 1.0
    require_replay_binding: bool = False

    def __post_init__(self) -> None:
        for name in (
            "minimum_retained_prior_mass",
            "maximum_omitted_posterior_mass",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real probability")
            normalized = float(value)
            if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, normalized)
        if type(self.require_replay_binding) is not bool:
            raise ValueError("require_replay_binding must be a Boolean")


class MultiContactInferenceRejectedError(ValueError):
    """Raised when an inference bank fails a configured admission policy."""


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
    metadata: Mapping[str, Any]

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
        metadata = validated_json_mapping(
            self.metadata,
            error_message="multi-contact posterior metadata must be finite JSON data",
        )
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
        object.__setattr__(self, "metadata", metadata)

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
    cumulative_command_energy = np.cumsum(np.sum(np.square(command_change), axis=0))
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


def _normalized_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("multi-contact posterior normalization failed")
    return weights / total


def _logsumexp(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    maximum = float(np.max(finite))
    return maximum + float(np.log(np.sum(np.exp(finite - maximum))))


def _omitted_posterior_mass_bound(
    *,
    retained_prior_mass: float,
    log_prior: np.ndarray,
    log_likelihood: np.ndarray,
    settings: DynamicContactInferenceConfig,
    dynamic_score_used: bool,
) -> tuple[float, float, float]:
    retained_log_average_likelihood = _logsumexp(log_prior + log_likelihood)
    maximum_score = -float(np.log(settings.observation_scale_m))
    if dynamic_score_used:
        minimum_difference_scale = np.sqrt(2.0) * settings.observation_scale_m
        maximum_score += settings.dynamic_likelihood_weight * -float(
            np.log(minimum_difference_scale)
        )
    maximum_log_likelihood = settings.likelihood_power * maximum_score
    if retained_prior_mass >= 1.0:
        return 0.0, retained_log_average_likelihood, maximum_log_likelihood
    retained_log_evidence = (
        float(np.log(retained_prior_mass)) + retained_log_average_likelihood
    )
    omitted_log_evidence_bound = (
        float(np.log1p(-retained_prior_mass)) + maximum_log_likelihood
    )
    posterior_bound = float(
        np.exp(
            omitted_log_evidence_bound
            - np.logaddexp(retained_log_evidence, omitted_log_evidence_bound)
        )
    )
    return (
        min(max(posterior_bound, 0.0), 1.0),
        retained_log_average_likelihood,
        maximum_log_likelihood,
    )


def _gaussian_mixture_quantile(
    component_means: np.ndarray,
    component_variances: np.ndarray,
    weights: np.ndarray,
    probability: float,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    if not 0.0 < probability < 1.0:
        raise ValueError("mixture quantile probability must lie in (0, 1)")
    means = np.asarray(component_means, dtype=float)
    variances = np.asarray(component_variances, dtype=float)
    if means.shape != variances.shape or means.ndim < 2:
        raise ValueError("mixture means and variances must share shape (K, ...)")
    flat_means = means.reshape(means.shape[0], -1)
    flat_scales = np.sqrt(
        np.maximum(variances.reshape(variances.shape[0], -1), np.finfo(float).tiny)
    )
    result = np.empty(flat_means.shape[1], dtype=float)
    for start in range(0, flat_means.shape[1], chunk_size):
        stop = min(start + chunk_size, flat_means.shape[1])
        means_chunk = flat_means[:, start:stop]
        scales_chunk = flat_scales[:, start:stop]
        lower = np.min(means_chunk - 12.0 * scales_chunk, axis=0)
        upper = np.max(means_chunk + 12.0 * scales_chunk, axis=0)
        for _ in range(64):
            midpoint = 0.5 * (lower + upper)
            cdf = np.einsum(
                "k,km->m",
                weights,
                ndtr((midpoint[None] - means_chunk) / scales_chunk),
            )
            lower = np.where(cdf < probability, midpoint, lower)
            upper = np.where(cdf < probability, upper, midpoint)
        result[start:stop] = 0.5 * (lower + upper)
    return result.reshape(means.shape[1:])


def _mixture_interval(
    component_means: np.ndarray,
    component_variances: np.ndarray,
    weights: np.ndarray,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    tail = 0.5 * (1.0 - confidence_level)
    return (
        _gaussian_mixture_quantile(
            component_means,
            component_variances,
            weights,
            tail,
        ),
        _gaussian_mixture_quantile(
            component_means,
            component_variances,
            weights,
            1.0 - tail,
        ),
    )


def _infer_multi_contact_core(
    bank: MultiContactPathBank,
    observations_m: np.ndarray,
    *,
    prefix_frame_count: int,
    command_activation: np.ndarray,
    mask: np.ndarray | None,
    ood_distance: np.ndarray | None,
    settings: DynamicContactInferenceConfig,
) -> MultiContactPosterior:
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
        settings.observation_scale_m**2 + conditional_variance[:, :prefix_frame_count]
    )
    position_score = student_t_mean_log_score(
        bank.trajectories_m[:, :prefix_frame_count] - prefix_observations[None],
        prefix_valid,
        scale=position_scale,
        degrees_of_freedom=settings.degrees_of_freedom,
        reduction_axes=(1, 2, 3),
        empty_error="multi-contact update has no valid coordinates",
    )
    score = position_score
    dynamic_score_used = False
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
            dynamic_score = student_t_mean_log_score(
                predicted_delta - observed_delta[None],
                delta_valid,
                scale=np.sqrt(delta_variance),
                degrees_of_freedom=settings.degrees_of_freedom,
                reduction_axes=(1, 2, 3),
                empty_error="multi-contact dynamic update has no valid coordinates",
            )
            score = score + settings.dynamic_likelihood_weight * dynamic_score
            dynamic_score_used = True

    log_prior = log_weights_from_probabilities(
        bank.prior_weights,
        name="multi-contact prior",
    )
    log_likelihood = settings.likelihood_power * score
    weights = _normalized_log_weights(log_prior + log_likelihood)
    omitted_bound, retained_log_likelihood, maximum_log_likelihood = (
        _omitted_posterior_mass_bound(
            retained_prior_mass=bank.retained_prior_mass,
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            settings=settings,
            dynamic_score_used=dynamic_score_used,
        )
    )

    components = bank.trajectories_m
    mean = np.einsum("k,ktnc->tnc", weights, components)
    centered = components - mean[None]
    epistemic = np.einsum("k,ktnc->tnc", weights, np.square(centered))
    conditional = np.einsum("k,ktnc->tnc", weights, conditional_variance)
    variance = np.maximum(epistemic + conditional, np.finfo(float).tiny)
    lower, upper = _mixture_interval(
        components,
        conditional_variance,
        weights,
        settings.confidence_level,
    )

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
    switches[:, :, 1:] = bank.regime_paths[:, :, 1:] != bank.regime_paths[:, :, :-1]
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
        interval_lower_m=lower,
        interval_upper_m=upper,
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
            "omitted_posterior_mass_upper_bound": omitted_bound,
            "retained_log_average_likelihood": retained_log_likelihood,
            "maximum_omitted_log_likelihood": maximum_log_likelihood,
            "schedule_identity": bank.schedule_identity,
            "schedule_schema_version": MULTI_CONTACT_SCHEDULE_SCHEMA_VERSION,
            "rollout_identity": bank.rollout_identity,
            "rollout_schema_version": MULTI_CONTACT_ROLLOUT_SCHEMA_VERSION,
            "replay_result_identity": bank.replay_result_identity,
            "replay_bound": bank.replay_bound,
            "frame_times_s": (
                None
                if bank.frame_times_s is None
                else np.asarray(bank.frame_times_s).tolist()
            ),
            "interval_method": "conditional_gaussian_mixture_quantiles",
            "config": asdict(settings),
        },
    )


def _policy_violations(
    posterior: MultiContactPosterior,
    bank: MultiContactPathBank,
    policy: MultiContactInferencePolicy,
) -> list[str]:
    violations: list[str] = []
    if bank.retained_prior_mass + 1e-15 < policy.minimum_retained_prior_mass:
        violations.append("retained_prior_mass_below_minimum")
    omitted_bound = float(posterior.metadata["omitted_posterior_mass_upper_bound"])
    if omitted_bound > policy.maximum_omitted_posterior_mass + 1e-15:
        violations.append("omitted_posterior_mass_above_maximum")
    if policy.require_replay_binding and not bank.replay_bound:
        violations.append("replay_binding_required")
    return violations


def _validate_static_fallback(
    candidate: MultiContactPathBank,
    fallback: MultiContactPathBank,
    policy: MultiContactInferencePolicy,
) -> None:
    if fallback.contact_ids != candidate.contact_ids:
        raise ValueError("static fallback must use the candidate contact identifiers")
    if fallback.trajectories_m.shape[1:] != candidate.trajectories_m.shape[1:]:
        raise ValueError("static fallback must match candidate rollout dimensions")
    if len(fallback.prior_weights) != 1:
        raise ValueError("static fallback must contain exactly one rollout path")
    if not np.array_equal(
        fallback.regime_paths[:, :, 1:],
        fallback.regime_paths[:, :, :-1],
    ):
        raise ValueError("static fallback contact regimes must not change over time")
    if not np.isclose(
        fallback.retained_prior_mass,
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("static fallback must retain unit prior mass")
    if (
        candidate.frame_times_s is not None
        and fallback.frame_times_s is not None
        and not np.array_equal(candidate.frame_times_s, fallback.frame_times_s)
    ):
        raise ValueError("static fallback must use the candidate rollout timebase")
    if policy.require_replay_binding and not fallback.replay_bound:
        raise ValueError("static fallback must satisfy the replay-binding policy")


def _with_policy_metadata(
    posterior: MultiContactPosterior,
    extra: Mapping[str, Any],
) -> MultiContactPosterior:
    metadata = plain_json(posterior.metadata)
    metadata.update(plain_json(extra))
    return MultiContactPosterior(
        contact_ids=posterior.contact_ids,
        path_ids=posterior.path_ids,
        weights=posterior.weights,
        mean_m=posterior.mean_m,
        variance_m2=posterior.variance_m2,
        interval_lower_m=posterior.interval_lower_m,
        interval_upper_m=posterior.interval_upper_m,
        conditional_variance_m2=posterior.conditional_variance_m2,
        regime_probabilities=posterior.regime_probabilities,
        switch_probability=posterior.switch_probability,
        any_switch_probability=posterior.any_switch_probability,
        evidence_frame_stop=posterior.evidence_frame_stop,
        metadata=metadata,
    )


def infer_multi_contact_posterior(
    bank: MultiContactPathBank,
    observations_m: np.ndarray,
    *,
    prefix_frame_count: int,
    command_activation: np.ndarray,
    mask: np.ndarray | None = None,
    ood_distance: np.ndarray | None = None,
    config: DynamicContactInferenceConfig | None = None,
    policy: MultiContactInferencePolicy | None = None,
    static_fallback_bank: MultiContactPathBank | None = None,
) -> MultiContactPosterior:
    """Infer from a prefix and fail closed when retained support is inadequate.

    The default policy preserves the development API. Claim-bearing callers can
    require retained prior mass, a conservative omitted-posterior upper bound,
    and a replay-bound rollout identity. A rejected candidate either raises or
    returns the supplied one-path static fallback exactly.
    """

    if not isinstance(bank, MultiContactPathBank):
        raise ValueError("bank must be a MultiContactPathBank")
    if config is not None and not isinstance(config, DynamicContactInferenceConfig):
        raise ValueError("config must be a DynamicContactInferenceConfig")
    if policy is not None and not isinstance(policy, MultiContactInferencePolicy):
        raise ValueError("policy must be a MultiContactInferencePolicy")
    settings = config or DynamicContactInferenceConfig()
    admission = policy or MultiContactInferencePolicy()
    candidate = _infer_multi_contact_core(
        bank,
        observations_m,
        prefix_frame_count=prefix_frame_count,
        command_activation=command_activation,
        mask=mask,
        ood_distance=ood_distance,
        settings=settings,
    )
    violations = _policy_violations(candidate, bank, admission)
    policy_record = {
        "minimum_retained_prior_mass": admission.minimum_retained_prior_mass,
        "maximum_omitted_posterior_mass": (admission.maximum_omitted_posterior_mass),
        "require_replay_binding": admission.require_replay_binding,
    }
    if not violations:
        return _with_policy_metadata(
            candidate,
            {
                "inference_policy": policy_record,
                "support_gate_passed": True,
                "support_gate_violations": [],
                "fallback_used": False,
            },
        )
    if static_fallback_bank is None:
        details = ", ".join(violations)
        raise MultiContactInferenceRejectedError(
            f"multi-contact inference policy rejected rollout bank: {details}"
        )
    if not isinstance(static_fallback_bank, MultiContactPathBank):
        raise ValueError("static_fallback_bank must be a MultiContactPathBank")
    _validate_static_fallback(bank, static_fallback_bank, admission)
    fallback = _infer_multi_contact_core(
        static_fallback_bank,
        observations_m,
        prefix_frame_count=prefix_frame_count,
        command_activation=command_activation,
        mask=mask,
        ood_distance=ood_distance,
        settings=settings,
    )
    return _with_policy_metadata(
        fallback,
        {
            "inference_policy": policy_record,
            "support_gate_passed": False,
            "support_gate_violations": violations,
            "fallback_used": True,
            "rejected_schedule_identity": bank.schedule_identity,
            "rejected_rollout_identity": bank.rollout_identity,
            "rejected_retained_prior_mass": bank.retained_prior_mass,
            "rejected_omitted_posterior_mass_upper_bound": candidate.metadata[
                "omitted_posterior_mass_upper_bound"
            ],
            "fallback_schedule_identity": static_fallback_bank.schedule_identity,
            "fallback_rollout_identity": static_fallback_bank.rollout_identity,
        },
    )
