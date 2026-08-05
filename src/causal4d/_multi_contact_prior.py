"""Factorized top-k prior enumeration for multiple contact channels."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import prod
from numbers import Real
from typing import Sequence

import numpy as np

from causal4d._multi_contact_common import (
    activation_matrix,
    integer_array,
    normalized_weights,
    probability_mass,
    real_array,
    readonly,
    schedule_identity,
    validate_identifiers,
    validated_contact_ids,
)
from causal4d.dynamic_contact import (
    CONTACT_REGIME_NAMES,
    ContactPathPrior,
    ContactRegime,
    ContactTransitionConfig,
    enumerate_contact_paths,
)


@dataclass(frozen=True)
class MultiContactEnumerationConfig:
    """Controls deterministic pruning of the joint contact-path support.

    ``minimum_joint_probability`` is measured in the original marginal prior,
    before retained joint weights are renormalized.
    """

    maximum_joint_paths: int = 128
    minimum_joint_probability: float = 0.0

    def __post_init__(self) -> None:
        if type(self.maximum_joint_paths) is not int or self.maximum_joint_paths < 1:
            raise ValueError("maximum_joint_paths must be a positive integer")
        if isinstance(
            self.minimum_joint_probability, (bool, np.bool_)
        ) or not isinstance(self.minimum_joint_probability, Real):
            raise ValueError("minimum_joint_probability must be a real probability")
        minimum = float(self.minimum_joint_probability)
        if not np.isfinite(minimum) or not 0.0 <= minimum < 1.0:
            raise ValueError("minimum_joint_probability must lie in [0, 1)")
        object.__setattr__(self, "minimum_joint_probability", minimum)


@dataclass(frozen=True)
class MultiContactPathPrior:
    """Finite top-k approximation to a factorized joint contact-path prior."""

    contact_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    regime_paths: np.ndarray
    weights: np.ndarray
    retained_prior_mass: float
    marginal_retained_prior_mass: np.ndarray
    marginal_path_indices: np.ndarray
    marginal_path_counts: np.ndarray

    def __post_init__(self) -> None:
        paths = integer_array(
            self.regime_paths,
            name="regime_paths",
            dtype=np.int8,
        )
        weights = normalized_weights(self.weights, name="multi-contact path weights")
        marginal_mass = readonly(
            real_array(
                self.marginal_retained_prior_mass,
                name="marginal_retained_prior_mass",
            )
        )
        indices = integer_array(
            self.marginal_path_indices,
            name="marginal_path_indices",
        )
        counts = integer_array(
            self.marginal_path_counts,
            name="marginal_path_counts",
        )
        if paths.ndim != 3 or paths.shape[0] != len(weights):
            raise ValueError("regime_paths must have shape (K, G, T)")
        path_count, contact_count, frame_count = paths.shape
        if contact_count < 1 or frame_count < 1:
            raise ValueError("contact paths must contain contacts and frames")
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
        if np.any(paths < 0) or np.any(paths >= len(CONTACT_REGIME_NAMES)):
            raise ValueError("regime_paths contain an unknown contact regime")
        if marginal_mass.shape != (contact_count,):
            raise ValueError("marginal retained mass must identify every contact")
        if np.any((marginal_mass <= 0.0) | (marginal_mass > 1.0 + 1e-12)):
            raise ValueError("marginal retained mass must lie in (0, 1]")
        if indices.shape != (path_count, contact_count):
            raise ValueError("marginal_path_indices must have shape (K, G)")
        if counts.shape != (contact_count,) or np.any(counts < 1):
            raise ValueError("marginal_path_counts must be positive per contact")
        if np.any(indices < 0) or np.any(indices >= counts[None, :]):
            raise ValueError("marginal path index lies outside its support")
        retained = probability_mass(
            self.retained_prior_mass,
            name="retained_prior_mass",
        )
        if retained > float(np.prod(marginal_mass)) + 1e-12:
            raise ValueError("joint retained mass exceeds marginal retained support")
        object.__setattr__(self, "contact_ids", contact_ids)
        object.__setattr__(self, "path_ids", path_ids)
        object.__setattr__(self, "regime_paths", paths)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "retained_prior_mass", retained)
        object.__setattr__(self, "marginal_retained_prior_mass", marginal_mass)
        object.__setattr__(self, "marginal_path_indices", indices)
        object.__setattr__(self, "marginal_path_counts", counts)

    @property
    def full_cartesian_path_count(self) -> int:
        """Number of paths before the joint top-k truncation."""

        return int(prod(int(value) for value in self.marginal_path_counts))

    @property
    def schedule_identity(self) -> str:
        """Content identity for the complete retained joint schedule support."""

        return schedule_identity(
            self.contact_ids,
            self.path_ids,
            self.regime_paths,
            self.weights,
            self.retained_prior_mass,
        )


def _transition_configs(
    values: ContactTransitionConfig | Sequence[ContactTransitionConfig] | None,
    contact_count: int,
) -> tuple[ContactTransitionConfig, ...]:
    if values is None:
        return tuple(ContactTransitionConfig() for _ in range(contact_count))
    if isinstance(values, ContactTransitionConfig):
        return (values,) * contact_count
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(
            "transition_configs must identify every contact channel"
        ) from error
    if len(result) != contact_count or not all(
        isinstance(value, ContactTransitionConfig) for value in result
    ):
        raise ValueError("transition_configs must identify every contact channel")
    return result


def _initial_probabilities(
    values: np.ndarray | None,
    contact_count: int,
) -> tuple[np.ndarray | None, ...]:
    if values is None:
        return (None,) * contact_count
    probabilities = real_array(values, name="initial_probabilities")
    regime_count = len(ContactRegime)
    if probabilities.shape == (regime_count,):
        probabilities = np.broadcast_to(
            probabilities[None, :],
            (contact_count, regime_count),
        )
    if probabilities.shape != (contact_count, regime_count):
        raise ValueError("initial_probabilities must have shape (4,) or (G, 4)")
    return tuple(probabilities[index] for index in range(contact_count))


def _sorted_marginal(prior: ContactPathPrior) -> ContactPathPrior:
    order = sorted(
        range(len(prior.weights)),
        key=lambda index: (-float(prior.weights[index]), prior.path_ids[index]),
    )
    return ContactPathPrior(
        path_ids=tuple(prior.path_ids[index] for index in order),
        regime_paths=np.asarray(prior.regime_paths)[order],
        weights=np.asarray(prior.weights)[order],
        retained_prior_mass=prior.retained_prior_mass,
    )


def enumerate_multi_contact_paths(
    command_activation: np.ndarray,
    *,
    contact_ids: Sequence[str] | None = None,
    transition_configs: ContactTransitionConfig
    | Sequence[ContactTransitionConfig]
    | None = None,
    initial_probabilities: np.ndarray | None = None,
    config: MultiContactEnumerationConfig | None = None,
) -> MultiContactPathPrior:
    """Enumerate top joint paths without constructing the Cartesian product.

    Contact chains are conditionally independent in the prior. Each marginal is
    first enumerated by :func:`enumerate_contact_paths`; a best-first heap then
    extracts the globally highest-probability products. The reported retained
    mass accounts for both marginal beam pruning and joint top-k pruning.
    """

    if config is not None and not isinstance(config, MultiContactEnumerationConfig):
        raise ValueError("config must be a MultiContactEnumerationConfig")
    settings = config or MultiContactEnumerationConfig()
    activation = activation_matrix(command_activation)
    contact_count = activation.shape[0]
    identifiers = validated_contact_ids(contact_ids, contact_count)
    transitions = _transition_configs(transition_configs, contact_count)
    initials = _initial_probabilities(initial_probabilities, contact_count)
    marginals = tuple(
        _sorted_marginal(
            enumerate_contact_paths(
                activation[index],
                config=transitions[index],
                initial_probabilities=initials[index],
            )
        )
        for index in range(contact_count)
    )
    log_weights = tuple(
        np.log(np.asarray(prior.weights, dtype=float)) for prior in marginals
    )
    marginal_mass = np.asarray(
        [prior.retained_prior_mass for prior in marginals],
        dtype=float,
    )
    log_marginal_support_mass = float(np.sum(np.log(marginal_mass)))
    initial_index = (0,) * contact_count
    initial_log_probability = float(sum(values[0] for values in log_weights))
    heap: list[tuple[float, tuple[int, ...]]] = [
        (-initial_log_probability, initial_index)
    ]
    visited = {initial_index}
    selected_indices: list[tuple[int, ...]] = []
    selected_log_weights: list[float] = []
    log_minimum = (
        float(np.log(settings.minimum_joint_probability))
        if settings.minimum_joint_probability > 0.0
        else -np.inf
    )

    while heap and len(selected_indices) < settings.maximum_joint_paths:
        negative_log_probability, indices = heapq.heappop(heap)
        log_probability = -negative_log_probability
        if log_probability + log_marginal_support_mass < log_minimum:
            break
        selected_indices.append(indices)
        selected_log_weights.append(log_probability)
        for contact_index in range(contact_count):
            next_path_index = indices[contact_index] + 1
            if next_path_index >= len(marginals[contact_index].weights):
                continue
            neighbor = list(indices)
            neighbor[contact_index] = next_path_index
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple in visited:
                continue
            visited.add(neighbor_tuple)
            neighbor_log_probability = float(
                sum(
                    log_weights[index][path_index]
                    for index, path_index in enumerate(neighbor_tuple)
                )
            )
            heapq.heappush(heap, (-neighbor_log_probability, neighbor_tuple))

    if not selected_indices:
        raise RuntimeError("joint contact-path pruning removed all probability mass")
    selected_logs = np.asarray(selected_log_weights, dtype=float)
    maximum = float(np.max(selected_logs))
    scaled_weights = np.exp(selected_logs - maximum)
    normalized_joint_weights = scaled_weights / np.sum(scaled_weights)
    selected_support_mass = float(np.exp(maximum) * np.sum(scaled_weights))
    retained_prior_mass = float(
        np.exp(log_marginal_support_mass) * selected_support_mass
    )
    index_array = np.asarray(selected_indices, dtype=np.int64)
    paths = np.stack(
        [
            np.stack(
                [
                    marginals[contact_index].regime_paths[path_index]
                    for contact_index, path_index in enumerate(indices)
                ],
                axis=0,
            )
            for indices in selected_indices
        ],
        axis=0,
    )
    path_ids = tuple(
        ";".join(
            f"{identifiers[contact_index]}="
            f"{marginals[contact_index].path_ids[path_index]}"
            for contact_index, path_index in enumerate(indices)
        )
        for indices in selected_indices
    )
    return MultiContactPathPrior(
        contact_ids=identifiers,
        path_ids=path_ids,
        regime_paths=paths,
        weights=normalized_joint_weights,
        retained_prior_mass=retained_prior_mass,
        marginal_retained_prior_mass=marginal_mass,
        marginal_path_indices=index_array,
        marginal_path_counts=np.asarray(
            [len(prior.weights) for prior in marginals],
            dtype=np.int64,
        ),
    )
