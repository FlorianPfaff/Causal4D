"""Fresh-panel diagnostic for correlated contact-prefix evidence.

The registered latent-contact estimator is left unchanged. This module compares its
exact prefix likelihood with predeclared temporal blocks, graph-distance node
blocks, source-estimated residual whitening, and a generalized-Bayes likelihood
rate. Every non-registered choice is selected on source topologies only and is
then evaluated once on an untouched seed panel.
"""

from __future__ import annotations

import csv
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import scipy

from causal4d.benchmark import CounterfactualBenchmarkConfig, Episode
from causal4d.contact_concentration_diagnostic import (
    _CalibrationCase,
    _calibration_cases,
    _fit_objects,
    scale_probability_weights,
)
from causal4d.contact_evaluation import FoldCalibration, _calibrate_fold
from causal4d.contact_inference import (
    ContactRolloutBank,
    ContactState,
    GraphContactHypothesisModel,
    LatentContactConfig,
    build_rollout_bank,
    fit_contact_prior,
    true_contact_state,
)
from causal4d.contact_metrics import contact_recovery_metrics
from causal4d.simulator import graph_adjacency
from causal4d.weighting import log_weights_from_probabilities

REGISTERED_POLICY = "registered_exact"
FRAME_BLOCK_POLICY = "temporal_frame_blocks"
NODE_BLOCK_POLICY = "graph_distance_node_blocks"
WHITENED_POLICY = "source_residual_whitening"
GENERALIZED_BAYES_POLICY = "generalized_bayes_learning_rate"


@dataclass(frozen=True)
class CorrelationDiagnosticConfig:
    """Predeclared candidate grids and joint decision tolerances."""

    frame_block_sizes: tuple[int, ...] = (2, 3, 4)
    whitening_shrinkages: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    generalized_bayes_rates: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    whitening_eigenvalue_floor: float = 1e-6
    minimum_shifted_brier_improvement: float = 0.005
    maximum_matched_brier_degradation: float = 0.010
    maximum_node_accuracy_degradation: float = 0.020
    maximum_coverage_degradation: float = 0.050
    maximum_trajectory_relative_degradation: float = 0.050
    maximum_trajectory_absolute_degradation_m: float = 0.00005

    def __post_init__(self) -> None:
        if not self.frame_block_sizes or any(
            type(value) is not int or value < 2 for value in self.frame_block_sizes
        ):
            raise ValueError(
                "frame_block_sizes must be unique integers of at least two"
            )
        if len(set(self.frame_block_sizes)) != len(self.frame_block_sizes):
            raise ValueError("frame_block_sizes must be unique")
        for name, values in (
            ("whitening_shrinkages", self.whitening_shrinkages),
            ("generalized_bayes_rates", self.generalized_bayes_rates),
        ):
            array = np.asarray(values, dtype=float)
            if (
                array.size == 0
                or not np.all(np.isfinite(array))
                or np.any(array <= 0.0)
                or np.any(array > 1.0)
                or len(set(map(float, array))) != len(array)
            ):
                raise ValueError(f"{name} must contain unique finite values in (0, 1]")
        for name, value in asdict(self).items():
            if name in {
                "frame_block_sizes",
                "whitening_shrinkages",
                "generalized_bayes_rates",
            }:
                continue
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.whitening_eigenvalue_floor <= 0.0:
            raise ValueError("whitening_eigenvalue_floor must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _PreparedCase:
    bank: ContactRolloutBank
    episode: Episode
    observations: np.ndarray
    prefix_frame_count: int
    position_residual: np.ndarray
    velocity_residual: np.ndarray
    acceleration_residual: np.ndarray
    exact_energy_m2: np.ndarray
    node_block_energy_m2: np.ndarray
    residual_features_m: np.ndarray
    node_block_count: int


def _validate_case_inputs(
    bank: ContactRolloutBank,
    observations: np.ndarray,
    prefix_frame_count: int,
) -> np.ndarray:
    values = np.asarray(observations, dtype=float)
    expected = (
        bank.action.frame_count,
        bank.graph_object.node_count,
        2,
    )
    if values.shape != expected:
        raise ValueError(f"observations must have shape {expected}")
    if not 2 <= prefix_frame_count < bank.action.frame_count:
        raise ValueError("prefix_frame_count must leave at least one future frame")
    if not np.all(np.isfinite(values[:prefix_frame_count])):
        raise ValueError("the observation prefix must be finite")
    return values


def _residual_components(
    bank: ContactRolloutBank,
    observations: np.ndarray,
    prefix_frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _validate_case_inputs(bank, observations, prefix_frame_count)
    predicted = bank.trajectories[:, :, 1:prefix_frame_count]
    observed = values[1:prefix_frame_count]
    position = predicted - observed[None, None, ...]
    velocity = np.diff(position, axis=2)
    acceleration = np.diff(velocity, axis=2)
    return position, velocity, acceleration


def _exact_energy_m2(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dynamic_likelihood_weight: float,
) -> np.ndarray:
    energy = np.sum(np.square(position), axis=(2, 3, 4))
    if dynamic_likelihood_weight:
        energy = energy + dynamic_likelihood_weight * (
            0.5 * np.sum(np.square(velocity), axis=(2, 3, 4))
            + np.sum(np.square(acceleration), axis=(2, 3, 4)) / 6.0
        )
    return energy


def _temporal_block_energy_m2(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dynamic_likelihood_weight: float,
    block_size: int,
) -> np.ndarray:
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    output = np.zeros(position.shape[:2], dtype=float)
    components = (
        (position, 1.0),
        (velocity, 0.5 * dynamic_likelihood_weight),
        (acceleration, dynamic_likelihood_weight / 6.0),
    )
    for residual, weight in components:
        if weight == 0.0 or residual.shape[2] == 0:
            continue
        per_frame = weight * np.sum(np.square(residual), axis=(3, 4))
        for start in range(0, residual.shape[2], block_size):
            output += np.mean(per_frame[:, :, start : start + block_size], axis=2)
    return output


def _graph_distance_node_groups(
    bank: ContactRolloutBank,
) -> tuple[tuple[int, ...], ...]:
    adjacency = graph_adjacency(bank.graph_object)
    distances = [math.inf] * bank.graph_object.node_count
    pending: deque[int] = deque()
    for node in bank.action.contact_nodes:
        distances[node] = 0
        pending.append(node)
    while pending:
        node = pending.popleft()
        next_distance = int(distances[node]) + 1
        for neighbour in adjacency[node]:
            neighbour = int(neighbour)
            if distances[neighbour] > next_distance:
                distances[neighbour] = next_distance
                pending.append(neighbour)
    finite = sorted({int(value) for value in distances if math.isfinite(value)})
    groups = [
        tuple(index for index, value in enumerate(distances) if value == distance)
        for distance in finite
    ]
    unreachable = tuple(
        index for index, value in enumerate(distances) if not math.isfinite(value)
    )
    if unreachable:
        groups.append(unreachable)
    if not groups or any(not group for group in groups):
        raise RuntimeError("graph-distance node partition is empty")
    return tuple(groups)


def _node_block_energy_m2(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dynamic_likelihood_weight: float,
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    output = np.zeros(position.shape[:2], dtype=float)
    components = (
        (position, 1.0),
        (velocity, 0.5 * dynamic_likelihood_weight),
        (acceleration, dynamic_likelihood_weight / 6.0),
    )
    for residual, weight in components:
        if weight == 0.0 or residual.shape[2] == 0:
            continue
        per_node = weight * np.sum(np.square(residual), axis=(2, 4))
        for group in groups:
            output += np.mean(per_node[:, :, tuple(group)], axis=2)
    return output


def _weighted_residual_features(
    residual: np.ndarray,
    dynamic_likelihood_weight: float,
    *,
    time_axis: int,
) -> np.ndarray:
    parts = [residual]
    if dynamic_likelihood_weight:
        velocity = np.diff(residual, axis=time_axis)
        velocity_padded = np.zeros_like(residual)
        velocity_slice = [slice(None)] * residual.ndim
        velocity_slice[time_axis] = slice(1, None)
        velocity_padded[tuple(velocity_slice)] = (
            math.sqrt(0.5 * dynamic_likelihood_weight) * velocity
        )
        acceleration = np.diff(velocity, axis=time_axis)
        acceleration_padded = np.zeros_like(residual)
        acceleration_slice = [slice(None)] * residual.ndim
        acceleration_slice[time_axis] = slice(2, None)
        acceleration_padded[tuple(acceleration_slice)] = (
            math.sqrt(dynamic_likelihood_weight / 6.0) * acceleration
        )
        parts.extend((velocity_padded, acceleration_padded))
    return np.concatenate(parts, axis=-1)


def _bank_residual_features_m(
    position_residual: np.ndarray,
    dynamic_likelihood_weight: float,
) -> np.ndarray:
    features = _weighted_residual_features(
        position_residual,
        dynamic_likelihood_weight,
        time_axis=2,
    )
    return features.reshape(
        features.shape[0],
        features.shape[1],
        -1,
        features.shape[-1],
    )


def _truth_proxy_state_index(
    bank: ContactRolloutBank,
    truth: ContactState,
) -> tuple[int, float]:
    candidates = [
        index
        for index, state in enumerate(bank.contact_states)
        if state.contact_nodes == truth.contact_nodes
    ]
    if not candidates:
        raise RuntimeError(
            "source truth contact nodes are absent from the rollout bank"
        )

    def distance(index: int) -> float:
        state = bank.contact_states[index]
        return float(
            ((state.gain_multiplier - truth.gain_multiplier) / 0.15) ** 2
            + ((state.delay_steps - truth.delay_steps) / 1.0) ** 2
            + ((state.slip_fraction - truth.slip_fraction) / 0.20) ** 2
            + ((state.rotation_radians - truth.rotation_radians) / np.deg2rad(8.0)) ** 2
        )

    selected = min(candidates, key=lambda index: (distance(index), index))
    return selected, distance(selected)


def _source_feature_matrix(
    cases: Sequence[_CalibrationCase],
    calibration: FoldCalibration,
    prefix_frame_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    rows: list[np.ndarray] = []
    proxy_distances: list[float] = []
    for case in cases:
        truth = true_contact_state(
            case.bank.graph_object,
            case.episode.action,
            case.episode.condition,
        )
        proxy_index, proxy_distance = _truth_proxy_state_index(case.bank, truth)
        proxy_distances.append(proxy_distance)
        trajectories = case.bank.trajectories[proxy_index]
        predictive_mean = np.sum(
            case.bank.parameter_weights[:, None, None, None] * trajectories,
            axis=0,
        )
        observations = _validate_case_inputs(
            case.bank,
            case.observations,
            prefix_frame_count,
        )
        residual = (
            predictive_mean[1:prefix_frame_count] - observations[1:prefix_frame_count]
        )
        features = _weighted_residual_features(
            residual,
            calibration.dynamic_likelihood_weight,
            time_axis=0,
        )
        rows.append(
            features.reshape(-1, features.shape[-1]) / calibration.likelihood_scale_m
        )
    matrix = np.concatenate(rows, axis=0)
    if matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise RuntimeError("source whitening matrix is insufficient or non-finite")
    return matrix, {
        "mean_truth_proxy_distance": float(np.mean(proxy_distances)),
        "maximum_truth_proxy_distance": float(np.max(proxy_distances)),
    }


def _source_correlation(
    matrix: np.ndarray,
    *,
    variance_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("source residual features must be a nontrivial matrix")
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    diagonal = np.maximum(np.diag(covariance), variance_floor)
    scale = np.sqrt(diagonal)
    correlation = covariance / np.outer(scale, scale)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues = np.linalg.eigvalsh(correlation)
    condition = float(np.linalg.cond(correlation))
    return correlation, {
        "source_sample_count": int(values.shape[0]),
        "feature_dimension": int(values.shape[1]),
        "empirical_min_eigenvalue": float(np.min(eigenvalues)),
        "empirical_max_eigenvalue": float(np.max(eigenvalues)),
        "empirical_condition_number": condition if np.isfinite(condition) else None,
        "variance_floor": float(variance_floor),
        "correlation_matrix": correlation.tolist(),
    }


def _shrunken_correlation_inverse(
    correlation: np.ndarray,
    shrinkage: float,
    *,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    value = float(shrinkage)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("shrinkage must be finite and in (0, 1]")
    matrix = (1.0 - value) * correlation + value * np.eye(correlation.shape[0])
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floored = np.maximum(eigenvalues, eigenvalue_floor)
    inverse = (eigenvectors * (1.0 / floored)) @ eigenvectors.T
    return inverse, {
        "shrinkage": value,
        "minimum_eigenvalue_before_floor": float(np.min(eigenvalues)),
        "maximum_eigenvalue_before_floor": float(np.max(eigenvalues)),
        "eigenvalue_floor": float(eigenvalue_floor),
    }


def _whitened_energy_m2(
    features_m: np.ndarray,
    inverse_correlation: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "...sc,cd,...sd->...",
        features_m,
        inverse_correlation,
        features_m,
        optimize=True,
    )


def _raw_posterior_from_energy(
    bank: ContactRolloutBank,
    energy_m2: np.ndarray,
    calibration: FoldCalibration,
    *,
    learning_rate: float = 1.0,
) -> np.ndarray:
    rate = float(learning_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    energy = np.asarray(energy_m2, dtype=float)
    if energy.shape != bank.prior_joint_weights.shape:
        raise ValueError("likelihood energy has the wrong joint-support shape")
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("likelihood energy must be finite and nonnegative")
    log_weights = log_weights_from_probabilities(
        bank.prior_joint_weights,
        name="correlation-diagnostic prior weights",
    )
    log_weights -= (
        0.5
        * calibration.likelihood_power
        * rate
        * energy
        / calibration.likelihood_scale_m**2
    )
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    return weights


def _registered_weights(
    prepared: _PreparedCase,
    calibration: FoldCalibration,
) -> np.ndarray:
    reference = prepared.bank.update_weights(
        prepared.observations,
        prefix_frame_count=prepared.prefix_frame_count,
        likelihood_scale_m=calibration.likelihood_scale_m,
        likelihood_power=calibration.likelihood_power,
        dynamic_likelihood_weight=calibration.dynamic_likelihood_weight,
    )
    recomputed = _raw_posterior_from_energy(
        prepared.bank,
        prepared.exact_energy_m2,
        calibration,
    )
    if not np.allclose(reference, recomputed, rtol=1e-12, atol=1e-15):
        raise RuntimeError("registered prefix likelihood was not reproduced")
    return scale_probability_weights(reference, calibration.posterior_temperature)


def _prepare_case(
    case: _CalibrationCase,
    calibration: FoldCalibration,
    prefix_frame_count: int,
) -> _PreparedCase:
    position, velocity, acceleration = _residual_components(
        case.bank,
        case.observations,
        prefix_frame_count,
    )
    groups = _graph_distance_node_groups(case.bank)
    exact = _exact_energy_m2(
        position,
        velocity,
        acceleration,
        calibration.dynamic_likelihood_weight,
    )
    node_energy = _node_block_energy_m2(
        position,
        velocity,
        acceleration,
        calibration.dynamic_likelihood_weight,
        groups,
    )
    return _PreparedCase(
        bank=case.bank,
        episode=case.episode,
        observations=np.asarray(case.observations, dtype=float),
        prefix_frame_count=prefix_frame_count,
        position_residual=position,
        velocity_residual=velocity,
        acceleration_residual=acceleration,
        exact_energy_m2=exact,
        node_block_energy_m2=node_energy,
        residual_features_m=_bank_residual_features_m(
            position,
            calibration.dynamic_likelihood_weight,
        ),
        node_block_count=len(groups),
    )


def _posterior_entropy(probabilities: np.ndarray) -> tuple[float, float]:
    values = np.asarray(probabilities, dtype=float)
    positive = values > 0.0
    entropy = float(-np.sum(values[positive] * np.log(values[positive])))
    return entropy, float(np.exp(entropy))


def _candidate_descriptor(policy: str, value: float | int | None) -> dict[str, Any]:
    if policy == REGISTERED_POLICY:
        return {"policy": policy}
    if policy == FRAME_BLOCK_POLICY:
        return {"policy": policy, "frame_block_size": int(value)}
    if policy == NODE_BLOCK_POLICY:
        return {"policy": policy, "node_partition": "distance_from_nominal_contact"}
    if policy == WHITENED_POLICY:
        return {"policy": policy, "whitening_shrinkage": float(value)}
    if policy == GENERALIZED_BAYES_POLICY:
        return {"policy": policy, "learning_rate": float(value)}
    raise KeyError(policy)


def _candidate_weights(
    prepared: _PreparedCase,
    calibration: FoldCalibration,
    descriptor: Mapping[str, Any],
    *,
    whitening_inverses: Mapping[float, np.ndarray],
) -> np.ndarray:
    policy = str(descriptor["policy"])
    if policy == REGISTERED_POLICY:
        return _registered_weights(prepared, calibration)
    if policy == FRAME_BLOCK_POLICY:
        energy = _temporal_block_energy_m2(
            prepared.position_residual,
            prepared.velocity_residual,
            prepared.acceleration_residual,
            calibration.dynamic_likelihood_weight,
            int(descriptor["frame_block_size"]),
        )
        raw = _raw_posterior_from_energy(prepared.bank, energy, calibration)
    elif policy == NODE_BLOCK_POLICY:
        raw = _raw_posterior_from_energy(
            prepared.bank,
            prepared.node_block_energy_m2,
            calibration,
        )
    elif policy == WHITENED_POLICY:
        shrinkage = float(descriptor["whitening_shrinkage"])
        energy = _whitened_energy_m2(
            prepared.residual_features_m,
            whitening_inverses[shrinkage],
        )
        raw = _raw_posterior_from_energy(prepared.bank, energy, calibration)
    elif policy == GENERALIZED_BAYES_POLICY:
        raw = _raw_posterior_from_energy(
            prepared.bank,
            prepared.exact_energy_m2,
            calibration,
            learning_rate=float(descriptor["learning_rate"]),
        )
    else:
        raise KeyError(policy)
    return scale_probability_weights(raw, calibration.posterior_temperature)


def _effective_block_counts(
    prepared: _PreparedCase,
    calibration: FoldCalibration,
    descriptor: Mapping[str, Any],
) -> dict[str, int]:
    lengths = [prepared.position_residual.shape[2]]
    if calibration.dynamic_likelihood_weight:
        lengths.extend(
            (
                prepared.velocity_residual.shape[2],
                prepared.acceleration_residual.shape[2],
            )
        )
    node_count = prepared.bank.graph_object.node_count
    policy = str(descriptor["policy"])
    if policy == FRAME_BLOCK_POLICY:
        size = int(descriptor["frame_block_size"])
        temporal = sum(math.ceil(length / size) for length in lengths if length)
        residual_blocks = temporal * node_count
    elif policy == NODE_BLOCK_POLICY:
        temporal = len(lengths)
        residual_blocks = temporal * prepared.node_block_count
    elif policy == WHITENED_POLICY:
        temporal = prepared.position_residual.shape[2]
        residual_blocks = temporal * node_count
    else:
        temporal = sum(lengths)
        residual_blocks = temporal * node_count
    return {
        "effective_temporal_blocks": int(temporal),
        "effective_node_blocks": int(prepared.node_block_count),
        "effective_residual_blocks": int(residual_blocks),
    }


def _score_case(
    prepared: _PreparedCase,
    weights: np.ndarray,
    contact_config: LatentContactConfig,
    descriptor: Mapping[str, Any],
    calibration: FoldCalibration,
) -> dict[str, Any]:
    marginal = prepared.bank.contact_marginal(weights)
    truth = true_contact_state(
        prepared.bank.graph_object,
        prepared.episode.action,
        prepared.episode.condition,
    )
    recovery = contact_recovery_metrics(
        prepared.bank.contact_states,
        marginal,
        truth,
        confidence_level=contact_config.confidence_level,
    )
    prediction = prepared.bank.predictive_distribution(
        weights,
        method=f"correlation_diagnostic_{descriptor['policy']}",
        include_intervals=False,
    )
    prefix = prepared.prefix_frame_count
    trajectory_rmse = float(
        np.sqrt(
            np.mean(
                np.square(prediction.mean[prefix:] - prepared.episode.truth[prefix:])
            )
        )
    )
    truth_probability = float(recovery["node_truth_probability"])
    entropy, effective_support = _posterior_entropy(marginal)
    return {
        "node_correct": float(recovery["node_correct"]),
        "node_confidence": float(recovery["node_confidence"]),
        "node_truth_probability": truth_probability,
        "node_log_score": float(-np.log(max(truth_probability, np.finfo(float).tiny))),
        "node_brier": float(recovery["node_brier"]),
        "node_credible_covered": float(recovery["node_credible_covered"]),
        "posterior_entropy_nats": entropy,
        "posterior_effective_support": effective_support,
        "trajectory_rmse_m": trajectory_rmse,
        **_effective_block_counts(prepared, calibration, descriptor),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty correlation panel")
    accuracy = float(np.mean([float(row["node_correct"]) for row in rows]))
    confidence = float(np.mean([float(row["node_confidence"]) for row in rows]))
    return {
        "case_count": len(rows),
        "node_accuracy": accuracy,
        "mean_node_confidence": confidence,
        "node_calibration_error": abs(confidence - accuracy),
        "mean_node_truth_probability": float(
            np.mean([float(row["node_truth_probability"]) for row in rows])
        ),
        "mean_node_log_score": float(
            np.mean([float(row["node_log_score"]) for row in rows])
        ),
        "mean_node_brier": float(np.mean([float(row["node_brier"]) for row in rows])),
        "node_credible_coverage": float(
            np.mean([float(row["node_credible_covered"]) for row in rows])
        ),
        "mean_posterior_entropy_nats": float(
            np.mean([float(row["posterior_entropy_nats"]) for row in rows])
        ),
        "mean_posterior_effective_support": float(
            np.mean([float(row["posterior_effective_support"]) for row in rows])
        ),
        "mean_trajectory_rmse_m": float(
            np.mean([float(row["trajectory_rmse_m"]) for row in rows])
        ),
        "mean_effective_residual_blocks": float(
            np.mean([float(row["effective_residual_blocks"]) for row in rows])
        ),
    }


def _grouped_aggregates(
    rows: Sequence[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    keys = sorted({tuple(str(row[field]) for field in fields) for row in rows})
    output: list[dict[str, Any]] = []
    for key in keys:
        selected = [
            row for row in rows if tuple(str(row[field]) for field in fields) == key
        ]
        output.append({**dict(zip(fields, key, strict=True)), **_aggregate(selected)})
    return output


def _candidate_scores(
    prepared_cases: Sequence[_PreparedCase],
    descriptors: Sequence[Mapping[str, Any]],
    calibration: FoldCalibration,
    contact_config: LatentContactConfig,
    *,
    whitening_inverses: Mapping[float, np.ndarray],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for order, descriptor in enumerate(descriptors):
        rows = [
            _score_case(
                prepared,
                _candidate_weights(
                    prepared,
                    calibration,
                    descriptor,
                    whitening_inverses=whitening_inverses,
                ),
                contact_config,
                descriptor,
                calibration,
            )
            for prepared in prepared_cases
        ]
        output.append(
            {
                "candidate_order": order,
                "candidate": dict(descriptor),
                **_aggregate(rows),
            }
        )
    return output


def _select_candidate(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not scores:
        raise ValueError("policy selection requires candidate scores")
    selected = min(
        scores,
        key=lambda row: (
            float(row["mean_node_brier"]),
            float(row["mean_trajectory_rmse_m"]),
            int(row["candidate_order"]),
        ),
    )
    candidate = selected.get("candidate")
    if not isinstance(candidate, Mapping):
        raise RuntimeError("selected candidate descriptor is invalid")
    return dict(candidate)


def _policy_descriptors(
    config: CorrelationDiagnosticConfig,
) -> dict[str, tuple[dict[str, Any], ...]]:
    return {
        REGISTERED_POLICY: (_candidate_descriptor(REGISTERED_POLICY, None),),
        FRAME_BLOCK_POLICY: tuple(
            _candidate_descriptor(FRAME_BLOCK_POLICY, value)
            for value in config.frame_block_sizes
        ),
        NODE_BLOCK_POLICY: (_candidate_descriptor(NODE_BLOCK_POLICY, None),),
        WHITENED_POLICY: tuple(
            _candidate_descriptor(WHITENED_POLICY, value)
            for value in config.whitening_shrinkages
        ),
        GENERALIZED_BAYES_POLICY: tuple(
            _candidate_descriptor(GENERALIZED_BAYES_POLICY, value)
            for value in config.generalized_bayes_rates
        ),
    }


def _comparison_rows(
    aggregate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    policies = sorted(
        {str(row["policy"]) for row in aggregate if row["policy"] != REGISTERED_POLICY}
    )
    worlds = sorted({str(row["world_condition"]) for row in aggregate})
    output: list[dict[str, Any]] = []
    for policy in policies:
        for world in worlds:
            registered = next(
                row
                for row in aggregate
                if row["policy"] == REGISTERED_POLICY
                and row["world_condition"] == world
            )
            candidate = next(
                row
                for row in aggregate
                if row["policy"] == policy and row["world_condition"] == world
            )
            output.append(
                {
                    "policy": policy,
                    "world_condition": world,
                    "candidate_minus_registered_node_accuracy": (
                        candidate["node_accuracy"] - registered["node_accuracy"]
                    ),
                    "candidate_minus_registered_calibration_error": (
                        candidate["node_calibration_error"]
                        - registered["node_calibration_error"]
                    ),
                    "candidate_minus_registered_mean_brier": (
                        candidate["mean_node_brier"] - registered["mean_node_brier"]
                    ),
                    "candidate_minus_registered_mean_log_score": (
                        candidate["mean_node_log_score"]
                        - registered["mean_node_log_score"]
                    ),
                    "candidate_minus_registered_credible_coverage": (
                        candidate["node_credible_coverage"]
                        - registered["node_credible_coverage"]
                    ),
                    "candidate_minus_registered_trajectory_rmse_m": (
                        candidate["mean_trajectory_rmse_m"]
                        - registered["mean_trajectory_rmse_m"]
                    ),
                    "candidate_minus_registered_entropy_nats": (
                        candidate["mean_posterior_entropy_nats"]
                        - registered["mean_posterior_entropy_nats"]
                    ),
                    "candidate_minus_registered_effective_support": (
                        candidate["mean_posterior_effective_support"]
                        - registered["mean_posterior_effective_support"]
                    ),
                }
            )
    return output


def _decision_rows(
    aggregate: Sequence[Mapping[str, Any]],
    config: CorrelationDiagnosticConfig,
) -> list[dict[str, Any]]:
    policies = sorted(
        {str(row["policy"]) for row in aggregate if row["policy"] != REGISTERED_POLICY}
    )

    def row(policy: str, world: str) -> Mapping[str, Any]:
        return next(
            item
            for item in aggregate
            if item["policy"] == policy and item["world_condition"] == world
        )

    registered_matched = row(REGISTERED_POLICY, "matched_contact")
    registered_shifted = row(REGISTERED_POLICY, "shifted_contact")
    output: list[dict[str, Any]] = []
    for policy in policies:
        matched = row(policy, "matched_contact")
        shifted = row(policy, "shifted_contact")
        shifted_brier_improved = bool(
            shifted["mean_node_brier"]
            <= registered_shifted["mean_node_brier"]
            - config.minimum_shifted_brier_improvement
        )
        matched_brier_preserved = bool(
            matched["mean_node_brier"]
            <= registered_matched["mean_node_brier"]
            + config.maximum_matched_brier_degradation
        )
        accuracy_preserved = bool(
            matched["node_accuracy"]
            >= registered_matched["node_accuracy"]
            - config.maximum_node_accuracy_degradation
            and shifted["node_accuracy"]
            >= registered_shifted["node_accuracy"]
            - config.maximum_node_accuracy_degradation
        )
        coverage_preserved = bool(
            matched["node_credible_coverage"]
            >= registered_matched["node_credible_coverage"]
            - config.maximum_coverage_degradation
            and shifted["node_credible_coverage"]
            >= registered_shifted["node_credible_coverage"]
            - config.maximum_coverage_degradation
        )

        def trajectory_preserved(
            candidate: Mapping[str, Any],
            registered: Mapping[str, Any],
        ) -> bool:
            tolerance = max(
                config.maximum_trajectory_absolute_degradation_m,
                config.maximum_trajectory_relative_degradation
                * float(registered["mean_trajectory_rmse_m"]),
            )
            return bool(
                candidate["mean_trajectory_rmse_m"]
                <= registered["mean_trajectory_rmse_m"] + tolerance
            )

        trajectory_ok = trajectory_preserved(
            matched, registered_matched
        ) and trajectory_preserved(shifted, registered_shifted)
        promotion_candidate = bool(
            shifted_brier_improved
            and matched_brier_preserved
            and accuracy_preserved
            and coverage_preserved
            and trajectory_ok
        )
        output.append(
            {
                "policy": policy,
                "shifted_brier_improved": shifted_brier_improved,
                "matched_brier_preserved": matched_brier_preserved,
                "node_accuracy_preserved": accuracy_preserved,
                "credible_coverage_preserved": coverage_preserved,
                "trajectory_rmse_preserved": trajectory_ok,
                "promotion_candidate": promotion_candidate,
                "interpretation": (
                    "candidate_for_new_method_and_new_untouched_panel"
                    if promotion_candidate
                    else "bounded_or_negative_result"
                ),
            }
        )
    return output


def _runtime_environment() -> dict[str, Any]:
    installed: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[str(name).lower()] = distribution.version
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "causal4d_version": metadata.version("causal4d"),
        "installed_distributions": dict(sorted(installed.items())),
        "github": {
            name.lower(): os.environ.get(name)
            for name in (
                "GITHUB_REPOSITORY",
                "GITHUB_SHA",
                "GITHUB_RUN_ID",
                "GITHUB_WORKFLOW",
                "RUNNER_NAME",
                "RUNNER_OS",
                "RUNNER_ARCH",
            )
        },
    }


def run_contact_correlation_diagnostic(
    seeds: Sequence[int],
    *,
    benchmark_config: CounterfactualBenchmarkConfig | None = None,
    contact_config: LatentContactConfig | None = None,
    diagnostic_config: CorrelationDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Run the predeclared source-only diagnostic on fresh held-out seeds."""

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be a nonempty unique sequence")
    if any(seed < 0 for seed in seed_values):
        raise ValueError("seeds must be nonnegative")
    benchmark = benchmark_config or CounterfactualBenchmarkConfig()
    contact = contact_config or LatentContactConfig()
    diagnostic = diagnostic_config or CorrelationDiagnosticConfig()
    policy_candidates = _policy_descriptors(diagnostic)

    evaluation_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    whitening_rows: list[dict[str, Any]] = []
    for seed in seed_values:
        fitted = _fit_objects(seed, benchmark)
        for target_index, target in enumerate(fitted):
            sources = tuple(
                item for index, item in enumerate(fitted) if index != target_index
            )
            source_names = ";".join(item.protocol.graph_object.name for item in sources)
            prior = fit_contact_prior(
                tuple(item.protocol for item in sources),
                contact,
                action_split="test",
            )
            model = GraphContactHypothesisModel(prior=prior, config=contact)
            calibration_seed = seed * 1_000_003 + target_index * 100_003 + 17
            calibration = _calibrate_fold(
                sources,
                model,
                benchmark,
                contact,
                calibration_seed=calibration_seed,
            )
            source_cases = _calibration_cases(
                sources,
                model,
                benchmark,
                contact,
                calibration_seed=calibration_seed,
            )
            prefix = contact.prefix_frame_count(benchmark.frame_count)
            prepared_sources = tuple(
                _prepare_case(case, calibration, prefix) for case in source_cases
            )
            feature_matrix, proxy_record = _source_feature_matrix(
                source_cases,
                calibration,
                prefix,
            )
            correlation, whitening_record = _source_correlation(
                feature_matrix,
                variance_floor=diagnostic.whitening_eigenvalue_floor,
            )
            whitening_inverses: dict[float, np.ndarray] = {}
            shrinkage_records: list[dict[str, float]] = []
            for shrinkage in diagnostic.whitening_shrinkages:
                inverse, record = _shrunken_correlation_inverse(
                    correlation,
                    shrinkage,
                    eigenvalue_floor=diagnostic.whitening_eigenvalue_floor,
                )
                whitening_inverses[float(shrinkage)] = inverse
                shrinkage_records.append(record)
            whitening_rows.append(
                {
                    "seed": seed,
                    "held_out_object": target.protocol.graph_object.name,
                    "source_objects": source_names,
                    "source_only": True,
                    "target_outcomes_read": False,
                    "source_sample_count": whitening_record["source_sample_count"],
                    "feature_dimension": whitening_record["feature_dimension"],
                    "empirical_min_eigenvalue": whitening_record[
                        "empirical_min_eigenvalue"
                    ],
                    "empirical_max_eigenvalue": whitening_record[
                        "empirical_max_eigenvalue"
                    ],
                    "empirical_condition_number": whitening_record[
                        "empirical_condition_number"
                    ],
                    "variance_floor": whitening_record["variance_floor"],
                    "mean_truth_proxy_distance": proxy_record[
                        "mean_truth_proxy_distance"
                    ],
                    "maximum_truth_proxy_distance": proxy_record[
                        "maximum_truth_proxy_distance"
                    ],
                    "correlation_matrix": json.dumps(
                        whitening_record["correlation_matrix"],
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "shrinkage_records": json.dumps(
                        shrinkage_records,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                }
            )

            selected: dict[str, dict[str, Any]] = {}
            for policy, descriptors in policy_candidates.items():
                scores = _candidate_scores(
                    prepared_sources,
                    descriptors,
                    calibration,
                    contact,
                    whitening_inverses=whitening_inverses,
                )
                selected[policy] = _select_candidate(scores)
                selection_rows.append(
                    {
                        "seed": seed,
                        "held_out_object": target.protocol.graph_object.name,
                        "source_objects": source_names,
                        "policy": policy,
                        "selected_candidate": json.dumps(
                            selected[policy],
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                        "source_only_selection": True,
                        "target_outcomes_read_during_selection": False,
                        "likelihood_scale_m": calibration.likelihood_scale_m,
                        "likelihood_power": calibration.likelihood_power,
                        "dynamic_likelihood_weight": (
                            calibration.dynamic_likelihood_weight
                        ),
                        "posterior_temperature": calibration.posterior_temperature,
                        "candidate_scores": json.dumps(
                            scores,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    }
                )

            bank = build_rollout_bank(
                target.protocol.graph_object,
                target.protocol.test_action,
                target.baselines.physics.posterior,
                model,
                simulator_config=benchmark.simulator,
                parameter_particle_count=contact.parameter_particle_count,
                variance_floor_m2=benchmark.predictive_variance_floor_m2,
                confidence_level=contact.confidence_level,
            )
            for condition_index, episode in enumerate(target.held_out):
                rng = np.random.default_rng(
                    seed * 1_000_003 + target_index * 10_007 + condition_index * 97
                )
                observations = episode.truth + rng.normal(
                    scale=contact.observation_noise_std_m,
                    size=episode.truth.shape,
                )
                prepared = _prepare_case(
                    _CalibrationCase(
                        bank=bank,
                        episode=episode,
                        observations=observations,
                    ),
                    calibration,
                    prefix,
                )
                for policy, descriptor in selected.items():
                    weights = _candidate_weights(
                        prepared,
                        calibration,
                        descriptor,
                        whitening_inverses=whitening_inverses,
                    )
                    score = _score_case(
                        prepared,
                        weights,
                        contact,
                        descriptor,
                        calibration,
                    )
                    evaluation_rows.append(
                        {
                            "seed": seed,
                            "object": target.protocol.graph_object.name,
                            "source_objects": source_names,
                            "world_condition": episode.condition.name,
                            "policy": policy,
                            "selected_candidate": json.dumps(
                                descriptor,
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                            "forecast_start_frame": prefix,
                            "likelihood_scale_m": calibration.likelihood_scale_m,
                            "likelihood_power": calibration.likelihood_power,
                            "dynamic_likelihood_weight": (
                                calibration.dynamic_likelihood_weight
                            ),
                            "posterior_temperature": (
                                calibration.posterior_temperature
                            ),
                            "residual_feature_dimension": int(
                                prepared.residual_features_m.shape[-1]
                            ),
                            **score,
                        }
                    )

    aggregate = _grouped_aggregates(
        evaluation_rows,
        ("policy", "world_condition"),
    )
    decisions = _decision_rows(aggregate, diagnostic)
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactCorrelationDiagnostic",
        "seeds": list(seed_values),
        "fresh_panel_boundary": {
            "evaluation_seeds": list(seed_values),
            "excluded_prior_panels": ["0:5", "100:120", "200:220"],
            "source_only_selection": True,
            "target_outcomes_read_during_selection": False,
        },
        "benchmark_config": benchmark.as_dict(),
        "contact_config": contact.as_dict(),
        "diagnostic_config": diagnostic.as_dict(),
        "policy_candidates": policy_candidates,
        "selection_rows": selection_rows,
        "whitening_rows": whitening_rows,
        "aggregate": aggregate,
        "by_topology": _grouped_aggregates(
            evaluation_rows,
            ("policy", "world_condition", "object"),
        ),
        "comparison": _comparison_rows(aggregate),
        "decision": decisions,
        "any_promotion_candidate": any(
            bool(row["promotion_candidate"]) for row in decisions
        ),
        "rows": evaluation_rows,
        "runtime_environment": _runtime_environment(),
        "claim_boundary": (
            "Exploratory fresh-panel correlation diagnostic only. The frozen "
            "estimator, registered likelihood, prior panels, thresholds, and "
            "36-execution physical protocol are unchanged. A passing candidate "
            "would require a new method version and another untouched panel."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty diagnostic artifact: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_contact_correlation_diagnostic(
    result: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Write summary, row-level evidence, selections, whitening data, and hashes."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "contact-correlation-diagnostic.json"
    rows_path = output / "contact-correlation-rows.csv"
    selection_path = output / "contact-correlation-selection.csv"
    whitening_path = output / "contact-correlation-whitening.csv"
    _write_json(
        summary_path,
        {
            key: value
            for key, value in result.items()
            if key not in {"rows", "selection_rows", "whitening_rows"}
        },
    )
    _write_csv(rows_path, result["rows"])
    _write_csv(selection_path, result["selection_rows"])
    _write_csv(whitening_path, result["whitening_rows"])
    payloads = (summary_path, rows_path, selection_path, whitening_path)
    manifest_path = output / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifact_kind": "Causal4DContactCorrelationDiagnosticManifest",
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in payloads
            },
        },
    )
    return {
        "summary": str(summary_path),
        "rows": str(rows_path),
        "selection": str(selection_path),
        "whitening": str(whitening_path),
        "manifest": str(manifest_path),
    }
