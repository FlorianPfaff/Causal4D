"""Untouched-panel diagnostic for topology-conditioned prefix covariance.

The completed ``300:320`` correlation study found one topology-local signal:
source-residual whitening improved shifted soft-block proper scores while harming
cloth and rope.  This module turns that observation into a separately versioned
hypothesis.  It uses the already-open ``300:320`` panel for nested development,
freezes one shared and one topology-conditioned covariance policy, and evaluates
those policies once on a disjoint untouched panel.

Nothing in this module changes the registered latent-contact estimator.  The
registered likelihood is always reproduced as an explicit baseline, and an
identity-shrinkage candidate gives every whitening policy an exact no-op option.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.immutable_array import readonly_array

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_concentration_diagnostic import (
    _CalibrationCase,
    _fit_objects,
    scale_probability_weights,
)
from causal4d.contact_correlation_diagnostic import (
    REGISTERED_POLICY,
    _PreparedCase,
    _aggregate,
    _prepare_case,
    _raw_posterior_from_energy,
    _registered_weights,
    _runtime_environment,
    _score_case,
    _shrunken_correlation_inverse,
    _source_correlation,
    _source_feature_matrix,
    _whitened_energy_m2,
)
from causal4d.contact_evaluation import FoldCalibration, _calibrate_fold
from causal4d.contact_inference import (
    GraphContactHypothesisModel,
    LatentContactConfig,
    build_rollout_bank,
    fit_contact_prior,
)

GLOBAL_POLICY = "development_global_residual_whitening"
TOPOLOGY_POLICY = "development_topology_residual_whitening"
_CANONICAL_RESIDUAL_FEATURE_DIMENSION = 6


@dataclass(frozen=True)
class TopologyCovarianceDiagnosticConfig:
    """Predeclared hierarchical covariance grid and joint decision rule."""

    shared_correlation_weights: tuple[float, ...] = (
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    )
    identity_shrinkages: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)
    eigenvalue_floor: float = 1e-6
    minimum_shifted_brier_improvement: float = 0.005
    minimum_shifted_brier_improvement_over_global: float = 0.002
    maximum_matched_brier_degradation: float = 0.010
    maximum_per_topology_shifted_brier_degradation: float = 0.010
    maximum_node_accuracy_degradation: float = 0.020
    maximum_coverage_degradation: float = 0.050
    maximum_trajectory_relative_degradation: float = 0.050
    maximum_trajectory_absolute_degradation_m: float = 0.00005

    def __post_init__(self) -> None:
        for name, values, allow_zero in (
            ("shared_correlation_weights", self.shared_correlation_weights, True),
            ("identity_shrinkages", self.identity_shrinkages, False),
        ):
            array = np.asarray(values, dtype=float)
            lower_invalid = array < 0.0 if allow_zero else array <= 0.0
            if (
                array.size == 0
                or not np.all(np.isfinite(array))
                or np.any(lower_invalid)
                or np.any(array > 1.0)
                or len(set(map(float, array))) != len(array)
            ):
                interval = "[0, 1]" if allow_zero else "(0, 1]"
                raise ValueError(
                    f"{name} must contain unique finite values in {interval}"
                )
        if 1.0 not in self.identity_shrinkages:
            raise ValueError(
                "identity_shrinkages must contain 1.0 as an exact no-op candidate"
            )
        for name, value in asdict(self).items():
            if name in {"shared_correlation_weights", "identity_shrinkages"}:
                continue
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.eigenvalue_floor <= 0.0:
            raise ValueError("eigenvalue_floor must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _TopologyFold:
    seed: int
    object_name: str
    source_objects: tuple[str, ...]
    calibration: FoldCalibration
    prepared_cases: tuple[_PreparedCase, ...]
    feature_matrix: np.ndarray | None
    proxy_record: Mapping[str, float] | None


def _canonical_residual_features(values: np.ndarray) -> np.ndarray:
    """Return fixed position/velocity/acceleration feature coordinates.

    The registered source calibration may set the dynamic likelihood weight to
    zero.  The shared diagnostic covariance nevertheless needs one stable
    coordinate system across folds.  In that case the absent velocity and
    acceleration coordinates are represented by exact zeros, so an identity
    covariance remains bit-for-bit equivalent to the registered energy.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim < 2:
        raise ValueError("residual features must have a sample and feature axis")
    dimension = int(array.shape[-1])
    if dimension == _CANONICAL_RESIDUAL_FEATURE_DIMENSION:
        return array
    if dimension != 2:
        raise ValueError(
            "residual features must contain position-only or canonical "
            "position/velocity/acceleration coordinates"
        )
    padding = np.zeros(
        (*array.shape[:-1], _CANONICAL_RESIDUAL_FEATURE_DIMENSION - dimension),
        dtype=float,
    )
    return np.concatenate((array, padding), axis=-1)


def _validate_seed_panel(
    development_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    development = tuple(int(seed) for seed in development_seeds)
    evaluation = tuple(int(seed) for seed in evaluation_seeds)
    for name, values in (("development", development), ("evaluation", evaluation)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} seeds must be a nonempty unique sequence")
        if any(seed < 0 for seed in values):
            raise ValueError(f"{name} seeds must be nonnegative")
    if set(development) & set(evaluation):
        raise ValueError("development and evaluation seeds must be disjoint")
    return development, evaluation


def _build_seed_folds(
    seed: int,
    benchmark: CounterfactualBenchmarkConfig,
    contact: LatentContactConfig,
    *,
    include_development_features: bool,
) -> tuple[_TopologyFold, ...]:
    fitted = _fit_objects(seed, benchmark)
    output: list[_TopologyFold] = []
    for target_index, target in enumerate(fitted):
        sources = tuple(
            item for index, item in enumerate(fitted) if index != target_index
        )
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
        prefix = contact.prefix_frame_count(benchmark.frame_count)
        raw_cases: list[_CalibrationCase] = []
        prepared_cases: list[_PreparedCase] = []
        for condition_index, episode in enumerate(target.held_out):
            rng = np.random.default_rng(
                seed * 1_000_003 + target_index * 10_007 + condition_index * 97
            )
            observations = episode.truth + rng.normal(
                scale=contact.observation_noise_std_m,
                size=episode.truth.shape,
            )
            case = _CalibrationCase(
                bank=bank,
                episode=episode,
                observations=observations,
            )
            raw_cases.append(case)
            prepared_cases.append(_prepare_case(case, calibration, prefix))
        feature_matrix: np.ndarray | None = None
        proxy_record: Mapping[str, float] | None = None
        if include_development_features:
            feature_matrix, proxy_record = _source_feature_matrix(
                tuple(raw_cases),
                calibration,
                prefix,
            )
            feature_matrix = _canonical_residual_features(feature_matrix)
            feature_matrix = readonly_array(feature_matrix)
        output.append(
            _TopologyFold(
                seed=seed,
                object_name=target.protocol.graph_object.name,
                source_objects=tuple(
                    item.protocol.graph_object.name for item in sources
                ),
                calibration=calibration,
                prepared_cases=tuple(prepared_cases),
                feature_matrix=feature_matrix,
                proxy_record=proxy_record,
            )
        )
    return tuple(output)


def _correlation_from_folds(
    folds: Sequence[_TopologyFold],
    *,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrices = [fold.feature_matrix for fold in folds]
    if not matrices or any(matrix is None for matrix in matrices):
        raise ValueError("correlation fitting requires development feature matrices")
    feature_dimensions = {
        int(matrix.shape[1]) for matrix in matrices if matrix is not None
    }
    if len(feature_dimensions) != 1:
        raise RuntimeError("development residual feature dimensions disagree")
    matrix = np.concatenate(
        [np.asarray(item, dtype=float) for item in matrices if item is not None],
        axis=0,
    )
    correlation, record = _source_correlation(
        matrix,
        variance_floor=eigenvalue_floor,
    )
    return correlation, {
        **record,
        "fold_count": len(folds),
        "seed_count": len({fold.seed for fold in folds}),
        "object_names": sorted({fold.object_name for fold in folds}),
    }


def _hierarchical_inverse(
    topology_correlation: np.ndarray,
    shared_correlation: np.ndarray,
    *,
    shared_weight: float,
    identity_shrinkage: float,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    topology = np.asarray(topology_correlation, dtype=float)
    shared = np.asarray(shared_correlation, dtype=float)
    if topology.shape != shared.shape or topology.ndim != 2:
        raise ValueError("topology and shared correlations must be aligned matrices")
    if topology.shape[0] != topology.shape[1]:
        raise ValueError("correlation matrices must be square")
    weight = float(shared_weight)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("shared_weight must be finite and in [0, 1]")
    mixed = (1.0 - weight) * topology + weight * shared
    mixed = 0.5 * (mixed + mixed.T)
    np.fill_diagonal(mixed, 1.0)
    inverse, inverse_record = _shrunken_correlation_inverse(
        mixed,
        identity_shrinkage,
        eigenvalue_floor=eigenvalue_floor,
    )
    effective = (1.0 - float(identity_shrinkage)) * mixed + float(
        identity_shrinkage
    ) * np.eye(mixed.shape[0])
    return inverse, {
        "shared_correlation_weight": weight,
        "identity_shrinkage": float(identity_shrinkage),
        "mixed_correlation_matrix": mixed.tolist(),
        "effective_correlation_matrix": effective.tolist(),
        "inverse_correlation_matrix": inverse.tolist(),
        **inverse_record,
    }


def _candidate_descriptor(
    policy: str,
    *,
    shared_weight: float = 1.0,
    identity_shrinkage: float = 1.0,
) -> dict[str, Any]:
    if policy == REGISTERED_POLICY:
        return {"policy": REGISTERED_POLICY}
    if policy == GLOBAL_POLICY:
        return {
            "policy": GLOBAL_POLICY,
            "shared_correlation_weight": 1.0,
            "identity_shrinkage": float(identity_shrinkage),
        }
    if policy == TOPOLOGY_POLICY:
        return {
            "policy": TOPOLOGY_POLICY,
            "shared_correlation_weight": float(shared_weight),
            "identity_shrinkage": float(identity_shrinkage),
        }
    raise KeyError(policy)


def _candidate_weights(
    prepared: _PreparedCase,
    calibration: FoldCalibration,
    descriptor: Mapping[str, Any],
    *,
    inverse_correlation: np.ndarray | None,
) -> np.ndarray:
    policy = str(descriptor["policy"])
    if policy == REGISTERED_POLICY:
        if inverse_correlation is not None:
            raise ValueError("registered policy must not receive a covariance inverse")
        return _registered_weights(prepared, calibration)
    if policy not in {GLOBAL_POLICY, TOPOLOGY_POLICY}:
        raise KeyError(policy)
    if inverse_correlation is None:
        raise ValueError("whitening policies require a covariance inverse")
    residual_features = _canonical_residual_features(prepared.residual_features_m)
    energy = _whitened_energy_m2(
        residual_features,
        np.asarray(inverse_correlation, dtype=float),
    )
    raw = _raw_posterior_from_energy(prepared.bank, energy, calibration)
    return scale_probability_weights(raw, calibration.posterior_temperature)


def _score_fold(
    fold: _TopologyFold,
    descriptor: Mapping[str, Any],
    contact: LatentContactConfig,
    *,
    inverse_correlation: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prepared in fold.prepared_cases:
        weights = _candidate_weights(
            prepared,
            fold.calibration,
            descriptor,
            inverse_correlation=inverse_correlation,
        )
        rows.append(
            {
                "seed": fold.seed,
                "object": fold.object_name,
                "source_objects": ";".join(fold.source_objects),
                "world_condition": prepared.episode.condition.name,
                "policy": descriptor["policy"],
                "selected_candidate": json.dumps(
                    dict(descriptor),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "forecast_start_frame": prepared.prefix_frame_count,
                "likelihood_scale_m": fold.calibration.likelihood_scale_m,
                "likelihood_power": fold.calibration.likelihood_power,
                "dynamic_likelihood_weight": (
                    fold.calibration.dynamic_likelihood_weight
                ),
                "posterior_temperature": fold.calibration.posterior_temperature,
                "active_residual_feature_dimension": int(
                    prepared.residual_features_m.shape[-1]
                ),
                "residual_feature_dimension": (_CANONICAL_RESIDUAL_FEATURE_DIMENSION),
                **_score_case(
                    prepared,
                    weights,
                    contact,
                    descriptor,
                    fold.calibration,
                ),
            }
        )
    return rows


def _select_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate selection requires cross-validation rows")
    selected = min(
        rows,
        key=lambda row: (
            float(row["mean_node_brier"]),
            float(row["mean_trajectory_rmse_m"]),
            int(row["candidate_order"]),
        ),
    )
    descriptor = selected.get("candidate")
    if not isinstance(descriptor, Mapping):
        raise RuntimeError("selected covariance descriptor is invalid")
    return dict(descriptor)


def _global_cross_validation(
    folds: Sequence[_TopologyFold],
    config: TopologyCovarianceDiagnosticConfig,
    contact: LatentContactConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seeds = sorted({fold.seed for fold in folds})
    output: list[dict[str, Any]] = []
    for order, identity_shrinkage in enumerate(config.identity_shrinkages):
        descriptor = _candidate_descriptor(
            GLOBAL_POLICY,
            identity_shrinkage=identity_shrinkage,
        )
        scored: list[dict[str, Any]] = []
        for held_out_seed in seeds:
            training = [fold for fold in folds if fold.seed != held_out_seed]
            validation = [fold for fold in folds if fold.seed == held_out_seed]
            shared, _ = _correlation_from_folds(
                training,
                eigenvalue_floor=config.eigenvalue_floor,
            )
            inverse, _ = _hierarchical_inverse(
                shared,
                shared,
                shared_weight=1.0,
                identity_shrinkage=identity_shrinkage,
                eigenvalue_floor=config.eigenvalue_floor,
            )
            for fold in validation:
                scored.extend(
                    _score_fold(
                        fold,
                        descriptor,
                        contact,
                        inverse_correlation=inverse,
                    )
                )
        output.append(
            {
                "policy": GLOBAL_POLICY,
                "topology": "all",
                "candidate_order": order,
                "candidate": descriptor,
                "development_seed_count": len(seeds),
                "cross_validation": "leave_one_seed_out",
                **_aggregate(scored),
            }
        )
    return _select_candidate(output), output


def _topology_cross_validation(
    folds: Sequence[_TopologyFold],
    config: TopologyCovarianceDiagnosticConfig,
    contact: LatentContactConfig,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    seeds = sorted({fold.seed for fold in folds})
    topologies = sorted({fold.object_name for fold in folds})
    output: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for topology in topologies:
        topology_rows: list[dict[str, Any]] = []
        for order, (shared_weight, identity_shrinkage) in enumerate(
            product(
                config.shared_correlation_weights,
                config.identity_shrinkages,
            )
        ):
            descriptor = _candidate_descriptor(
                TOPOLOGY_POLICY,
                shared_weight=shared_weight,
                identity_shrinkage=identity_shrinkage,
            )
            scored: list[dict[str, Any]] = []
            for held_out_seed in seeds:
                training = [fold for fold in folds if fold.seed != held_out_seed]
                topology_training = [
                    fold for fold in training if fold.object_name == topology
                ]
                validation = [
                    fold
                    for fold in folds
                    if fold.seed == held_out_seed and fold.object_name == topology
                ]
                shared, _ = _correlation_from_folds(
                    training,
                    eigenvalue_floor=config.eigenvalue_floor,
                )
                topology_correlation, _ = _correlation_from_folds(
                    topology_training,
                    eigenvalue_floor=config.eigenvalue_floor,
                )
                inverse, _ = _hierarchical_inverse(
                    topology_correlation,
                    shared,
                    shared_weight=shared_weight,
                    identity_shrinkage=identity_shrinkage,
                    eigenvalue_floor=config.eigenvalue_floor,
                )
                for fold in validation:
                    scored.extend(
                        _score_fold(
                            fold,
                            descriptor,
                            contact,
                            inverse_correlation=inverse,
                        )
                    )
            topology_rows.append(
                {
                    "policy": TOPOLOGY_POLICY,
                    "topology": topology,
                    "candidate_order": order,
                    "candidate": descriptor,
                    "development_seed_count": len(seeds),
                    "cross_validation": "leave_one_seed_out",
                    **_aggregate(scored),
                }
            )
        selected[topology] = _select_candidate(topology_rows)
        output.extend(topology_rows)
    return selected, output


def _final_covariances(
    folds: Sequence[_TopologyFold],
    global_descriptor: Mapping[str, Any],
    topology_descriptors: Mapping[str, Mapping[str, Any]],
    config: TopologyCovarianceDiagnosticConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    shared, shared_record = _correlation_from_folds(
        folds,
        eigenvalue_floor=config.eigenvalue_floor,
    )
    global_inverse, global_record = _hierarchical_inverse(
        shared,
        shared,
        shared_weight=1.0,
        identity_shrinkage=float(global_descriptor["identity_shrinkage"]),
        eigenvalue_floor=config.eigenvalue_floor,
    )
    covariance_rows: list[dict[str, Any]] = [
        {
            "policy": GLOBAL_POLICY,
            "topology": "all",
            "selected_candidate": json.dumps(
                dict(global_descriptor),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "source_sample_count": shared_record["source_sample_count"],
            "development_fold_count": shared_record["fold_count"],
            "development_seed_count": shared_record["seed_count"],
            "feature_dimension": shared_record["feature_dimension"],
            "empirical_min_eigenvalue": shared_record["empirical_min_eigenvalue"],
            "empirical_max_eigenvalue": shared_record["empirical_max_eigenvalue"],
            "empirical_condition_number": shared_record["empirical_condition_number"],
            "empirical_correlation_matrix": json.dumps(
                shared_record["correlation_matrix"],
                separators=(",", ":"),
                allow_nan=False,
            ),
            "effective_correlation_matrix": json.dumps(
                global_record["effective_correlation_matrix"],
                separators=(",", ":"),
                allow_nan=False,
            ),
            "inverse_correlation_matrix": json.dumps(
                global_record["inverse_correlation_matrix"],
                separators=(",", ":"),
                allow_nan=False,
            ),
        }
    ]
    topology_inverses: dict[str, np.ndarray] = {}
    for topology in sorted(topology_descriptors):
        topology_folds = [fold for fold in folds if fold.object_name == topology]
        topology_correlation, topology_record = _correlation_from_folds(
            topology_folds,
            eigenvalue_floor=config.eigenvalue_floor,
        )
        descriptor = topology_descriptors[topology]
        inverse, record = _hierarchical_inverse(
            topology_correlation,
            shared,
            shared_weight=float(descriptor["shared_correlation_weight"]),
            identity_shrinkage=float(descriptor["identity_shrinkage"]),
            eigenvalue_floor=config.eigenvalue_floor,
        )
        topology_inverses[topology] = inverse
        proxy_records = [
            fold.proxy_record
            for fold in topology_folds
            if fold.proxy_record is not None
        ]
        covariance_rows.append(
            {
                "policy": TOPOLOGY_POLICY,
                "topology": topology,
                "selected_candidate": json.dumps(
                    dict(descriptor),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "source_sample_count": topology_record["source_sample_count"],
                "development_fold_count": topology_record["fold_count"],
                "development_seed_count": topology_record["seed_count"],
                "feature_dimension": topology_record["feature_dimension"],
                "empirical_min_eigenvalue": topology_record["empirical_min_eigenvalue"],
                "empirical_max_eigenvalue": topology_record["empirical_max_eigenvalue"],
                "empirical_condition_number": topology_record[
                    "empirical_condition_number"
                ],
                "mean_truth_proxy_distance": float(
                    np.mean(
                        [
                            float(item["mean_truth_proxy_distance"])
                            for item in proxy_records
                        ]
                    )
                ),
                "maximum_truth_proxy_distance": float(
                    np.max(
                        [
                            float(item["maximum_truth_proxy_distance"])
                            for item in proxy_records
                        ]
                    )
                ),
                "empirical_correlation_matrix": json.dumps(
                    topology_record["correlation_matrix"],
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "effective_correlation_matrix": json.dumps(
                    record["effective_correlation_matrix"],
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "inverse_correlation_matrix": json.dumps(
                    record["inverse_correlation_matrix"],
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
    return global_inverse, topology_inverses, covariance_rows


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


def _comparison_rows(
    aggregate: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    worlds = sorted({str(row["world_condition"]) for row in aggregate})
    for policy in (GLOBAL_POLICY, TOPOLOGY_POLICY):
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
                }
            )
    return output


def _decision_rows(
    aggregate: Sequence[Mapping[str, Any]],
    by_topology: Sequence[Mapping[str, Any]],
    config: TopologyCovarianceDiagnosticConfig,
) -> list[dict[str, Any]]:
    def aggregate_row(policy: str, world: str) -> Mapping[str, Any]:
        return next(
            row
            for row in aggregate
            if row["policy"] == policy and row["world_condition"] == world
        )

    registered_matched = aggregate_row(REGISTERED_POLICY, "matched_contact")
    registered_shifted = aggregate_row(REGISTERED_POLICY, "shifted_contact")
    global_shifted = aggregate_row(GLOBAL_POLICY, "shifted_contact")
    topologies = sorted({str(row["object"]) for row in by_topology})
    output: list[dict[str, Any]] = []
    for policy in (GLOBAL_POLICY, TOPOLOGY_POLICY):
        matched = aggregate_row(policy, "matched_contact")
        shifted = aggregate_row(policy, "shifted_contact")
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
            matched,
            registered_matched,
        ) and trajectory_preserved(shifted, registered_shifted)
        per_topology_preserved = True
        topology_effects: list[dict[str, Any]] = []
        for topology in topologies:
            registered_topology = next(
                row
                for row in by_topology
                if row["policy"] == REGISTERED_POLICY
                and row["world_condition"] == "shifted_contact"
                and row["object"] == topology
            )
            candidate_topology = next(
                row
                for row in by_topology
                if row["policy"] == policy
                and row["world_condition"] == "shifted_contact"
                and row["object"] == topology
            )
            delta = float(
                candidate_topology["mean_node_brier"]
                - registered_topology["mean_node_brier"]
            )
            preserved = bool(
                delta <= config.maximum_per_topology_shifted_brier_degradation
            )
            per_topology_preserved = per_topology_preserved and preserved
            topology_effects.append(
                {
                    "topology": topology,
                    "candidate_minus_registered_shifted_brier": delta,
                    "preserved": preserved,
                }
            )
        beats_global = (
            True
            if policy == GLOBAL_POLICY
            else bool(
                shifted["mean_node_brier"]
                <= global_shifted["mean_node_brier"]
                - config.minimum_shifted_brier_improvement_over_global
            )
        )
        promotion_candidate = bool(
            shifted_brier_improved
            and matched_brier_preserved
            and accuracy_preserved
            and coverage_preserved
            and trajectory_ok
            and per_topology_preserved
            and beats_global
        )
        output.append(
            {
                "policy": policy,
                "shifted_brier_improved": shifted_brier_improved,
                "matched_brier_preserved": matched_brier_preserved,
                "node_accuracy_preserved": accuracy_preserved,
                "credible_coverage_preserved": coverage_preserved,
                "trajectory_rmse_preserved": trajectory_ok,
                "per_topology_shifted_brier_preserved": per_topology_preserved,
                "shifted_brier_improved_over_global": beats_global,
                "topology_effects": topology_effects,
                "promotion_candidate": promotion_candidate,
                "interpretation": (
                    "candidate_for_separately_versioned_method"
                    if promotion_candidate
                    else "bounded_or_negative_result"
                ),
            }
        )
    return output


def run_contact_topology_covariance_diagnostic(
    development_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
    *,
    benchmark_config: CounterfactualBenchmarkConfig | None = None,
    contact_config: LatentContactConfig | None = None,
    diagnostic_config: TopologyCovarianceDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Fit on an opened panel and evaluate once on a disjoint seed panel."""

    development, evaluation = _validate_seed_panel(
        development_seeds,
        evaluation_seeds,
    )
    benchmark = benchmark_config or CounterfactualBenchmarkConfig()
    contact = contact_config or LatentContactConfig()
    diagnostic = diagnostic_config or TopologyCovarianceDiagnosticConfig()

    development_folds = tuple(
        fold
        for seed in development
        for fold in _build_seed_folds(
            seed,
            benchmark,
            contact,
            include_development_features=True,
        )
    )
    global_descriptor, global_cv = _global_cross_validation(
        development_folds,
        diagnostic,
        contact,
    )
    topology_descriptors, topology_cv = _topology_cross_validation(
        development_folds,
        diagnostic,
        contact,
    )
    global_inverse, topology_inverses, covariance_rows = _final_covariances(
        development_folds,
        global_descriptor,
        topology_descriptors,
        diagnostic,
    )

    evaluation_rows: list[dict[str, Any]] = []
    for seed in evaluation:
        for fold in _build_seed_folds(
            seed,
            benchmark,
            contact,
            include_development_features=False,
        ):
            evaluation_rows.extend(
                _score_fold(
                    fold,
                    _candidate_descriptor(REGISTERED_POLICY),
                    contact,
                    inverse_correlation=None,
                )
            )
            evaluation_rows.extend(
                _score_fold(
                    fold,
                    global_descriptor,
                    contact,
                    inverse_correlation=global_inverse,
                )
            )
            descriptor = topology_descriptors[fold.object_name]
            evaluation_rows.extend(
                _score_fold(
                    fold,
                    descriptor,
                    contact,
                    inverse_correlation=topology_inverses[fold.object_name],
                )
            )

    aggregate = _grouped_aggregates(
        evaluation_rows,
        ("policy", "world_condition"),
    )
    by_topology = _grouped_aggregates(
        evaluation_rows,
        ("policy", "world_condition", "object"),
    )
    decisions = _decision_rows(aggregate, by_topology, diagnostic)
    selection_rows = [*global_cv, *topology_cv]
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactTopologyCovarianceDiagnostic",
        "development_seeds": list(development),
        "evaluation_seeds": list(evaluation),
        "panel_boundary": {
            "opened_development_panel": list(development),
            "untouched_evaluation_panel": list(evaluation),
            "previously_opened_seed_ranges": [
                "0:5",
                "100:120",
                "200:220",
                "300:320",
            ],
            "development_outcomes_used_for_selection": True,
            "evaluation_outcomes_read_during_selection": False,
            "development_evaluation_disjoint": True,
        },
        "benchmark_config": benchmark.as_dict(),
        "contact_config": contact.as_dict(),
        "diagnostic_config": diagnostic.as_dict(),
        "selected_global_candidate": global_descriptor,
        "selected_topology_candidates": topology_descriptors,
        "aggregate": aggregate,
        "by_topology": by_topology,
        "comparison": _comparison_rows(aggregate),
        "decision": decisions,
        "any_promotion_candidate": any(
            bool(row["promotion_candidate"]) for row in decisions
        ),
        "topology_hypothesis_supported": next(
            bool(row["promotion_candidate"])
            for row in decisions
            if row["policy"] == TOPOLOGY_POLICY
        ),
        "selection_rows": selection_rows,
        "covariance_rows": covariance_rows,
        "rows": evaluation_rows,
        "runtime_environment": _runtime_environment(),
        "claim_boundary": (
            "Exploratory controlled topology-covariance diagnostic only. The "
            "registered estimator, frozen likelihood, prior result panels, exact-node "
            "gate, thresholds, and 36-execution physical protocol are unchanged. "
            "A passing result supports only a separately versioned method candidate."
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
    normalized = []
    for row in rows:
        normalized.append(
            {
                key: (
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )
    fieldnames: list[str] = []
    for row in normalized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)


def write_contact_topology_covariance_diagnostic(
    result: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Write summary, evaluation, selection, covariance, and manifest artifacts."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "contact-topology-covariance-diagnostic.json"
    rows_path = output / "contact-topology-covariance-rows.csv"
    selection_path = output / "contact-topology-covariance-selection.csv"
    covariance_path = output / "contact-topology-covariance-matrices.csv"
    _write_json(
        summary_path,
        {
            key: value
            for key, value in result.items()
            if key not in {"rows", "selection_rows", "covariance_rows"}
        },
    )
    _write_csv(rows_path, result["rows"])
    _write_csv(selection_path, result["selection_rows"])
    _write_csv(covariance_path, result["covariance_rows"])
    payloads = (summary_path, rows_path, selection_path, covariance_path)
    manifest_path = output / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifact_kind": ("Causal4DContactTopologyCovarianceDiagnosticManifest"),
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
        "covariance": str(covariance_path),
        "manifest": str(manifest_path),
    }
