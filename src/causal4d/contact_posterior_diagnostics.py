"""Topology-aware diagnostics for controlled realized-contact posteriors.

This module deliberately leaves the frozen estimator, benchmark gates, and primary
result-bundle schema unchanged. It recomputes the already registered controlled
posterior from the bundle's exact configuration, validates parity with the retained
contact-recovery rows, and writes a separate diagnostic artifact.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from itertools import permutations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.baselines import fit_baselines
from causal4d.benchmark import (
    CounterfactualBenchmarkConfig,
    build_protocol,
    generate_episodes,
    make_actions,
    make_parameter_grid,
)
from causal4d.contact_evaluation import (
    _FittedObject,
    _calibrate_fold,
    _temper_joint_weights,
)
from causal4d.contact_inference import (
    GraphContactHypothesisModel,
    LatentContactConfig,
    build_rollout_bank,
    fit_contact_prior,
    true_contact_state,
)
from causal4d.contact_metrics import contact_recovery_metrics
from causal4d.simulator import GraphObject


_FORCE_FIELD_THRESHOLDS = (0.80, 0.90, 0.95)
_FLOAT_PARITY_FIELDS = (
    "node_confidence",
    "node_truth_probability",
    "node_brier",
    "joint_effective_sample_size",
    "joint_normalized_entropy",
)
_EXACT_PARITY_FIELDS = (
    "node_truth",
    "node_map",
    "node_correct",
    "node_credible_covered",
    "delay_map",
    "delay_map_correct",
)


@dataclass(frozen=True)
class DiagnosticConfig:
    """Registered analysis-only choices for the contact diagnostic."""

    diffusion_strength: float = 1.0
    force_field_equivalence_threshold: float = 0.90
    parity_relative_tolerance: float = 2e-12
    parity_absolute_tolerance: float = 2e-15

    def __post_init__(self) -> None:
        if not np.isfinite(self.diffusion_strength) or self.diffusion_strength <= 0:
            raise ValueError("diffusion_strength must be finite and positive")
        if not 0.0 <= self.force_field_equivalence_threshold <= 1.0:
            raise ValueError(
                "force_field_equivalence_threshold must be in [0, 1]"
            )
        if (
            self.parity_relative_tolerance < 0.0
            or self.parity_absolute_tolerance < 0.0
        ):
            raise ValueError("parity tolerances must be nonnegative")

    def as_dict(self) -> dict[str, float]:
        return {
            "diffusion_strength": self.diffusion_strength,
            "force_field_equivalence_threshold": (
                self.force_field_equivalence_threshold
            ),
            "parity_relative_tolerance": self.parity_relative_tolerance,
            "parity_absolute_tolerance": self.parity_absolute_tolerance,
        }


def _node_label(nodes: Sequence[int]) -> str:
    return ";".join(map(str, nodes))


def _parse_nodes(value: str) -> tuple[int, ...]:
    stripped = value.strip()
    if not stripped:
        raise ValueError("contact-node label must be nonempty")
    return tuple(int(part) for part in stripped.split(";"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    raise ValueError(f"expected a serialized boolean, received {value!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty diagnostic artifact: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _adjacency(graph: GraphObject) -> tuple[tuple[int, ...], ...]:
    neighbors = [set() for _ in range(graph.rest_positions.shape[0])]
    for first, second in graph.edges:
        neighbors[int(first)].add(int(second))
        neighbors[int(second)].add(int(first))
    return tuple(tuple(sorted(values)) for values in neighbors)


def _all_pairs_graph_distances(graph: GraphObject) -> np.ndarray:
    adjacency = _adjacency(graph)
    count = len(adjacency)
    distances = np.full((count, count), count + 1, dtype=int)
    for source in range(count):
        distances[source, source] = 0
        frontier = [source]
        while frontier:
            node = frontier.pop(0)
            candidate_distance = distances[source, node] + 1
            for neighbor in adjacency[node]:
                if candidate_distance < distances[source, neighbor]:
                    distances[source, neighbor] = candidate_distance
                    frontier.append(neighbor)
    if np.any(distances > count):
        raise ValueError(f"graph {graph.name!r} is disconnected")
    return distances


def _assignment_distances(
    truth: tuple[int, ...],
    estimate: tuple[int, ...],
    distances: np.ndarray,
) -> tuple[float, int]:
    if len(truth) != len(estimate):
        raise ValueError("truth and MAP contact cardinalities differ")
    candidates: list[tuple[int, int]] = []
    for assignment in permutations(estimate):
        values = [
            int(distances[truth_node, estimate_node])
            for truth_node, estimate_node in zip(truth, assignment, strict=True)
        ]
        candidates.append((sum(values), max(values, default=0)))
    total, maximum = min(candidates)
    return total / max(len(truth), 1), maximum


def _node_signature(
    graph: GraphObject,
    node: int,
    distances: np.ndarray,
    *,
    include_sensors: bool,
) -> tuple[Any, ...]:
    adjacency = _adjacency(graph)
    degrees = tuple(len(values) for values in adjacency)
    signature: list[Any] = [
        degrees[node],
        tuple(sorted(degrees[neighbor] for neighbor in adjacency[node])),
        tuple(sorted(map(int, distances[node]))),
    ]
    if include_sensors:
        signature.append(
            tuple(sorted(int(distances[node, sensor]) for sensor in graph.sensor_nodes))
        )
    return tuple(signature)


def _assignment_signature(
    graph: GraphObject,
    nodes: tuple[int, ...],
    distances: np.ndarray,
    *,
    include_sensors: bool,
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                _node_signature(
                    graph,
                    node,
                    distances,
                    include_sensors=include_sensors,
                )
                for node in nodes
            ),
            key=repr,
        )
    )


def _diffused_force_field(
    graph: GraphObject,
    nodes: tuple[int, ...],
    *,
    strength: float,
) -> np.ndarray:
    count = graph.rest_positions.shape[0]
    adjacency_matrix = np.zeros((count, count), dtype=float)
    for first, second in graph.edges:
        adjacency_matrix[int(first), int(second)] = 1.0
        adjacency_matrix[int(second), int(first)] = 1.0
    laplacian = np.diag(np.sum(adjacency_matrix, axis=1)) - adjacency_matrix
    source = np.zeros(count, dtype=float)
    for node in nodes:
        source[node] += 1.0 / len(nodes)
    field = np.linalg.solve(
        np.eye(count, dtype=float) + strength * laplacian,
        source,
    )
    norm = float(np.linalg.norm(field))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("graph diffusion produced an invalid force-field proxy")
    return field / norm


def _posterior_diagnostic_metrics(
    states: Sequence[Any],
    weights: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, Any]:
    probabilities: dict[tuple[int, ...], float] = {}
    for state, weight in zip(states, weights, strict=True):
        nodes = tuple(map(int, state.contact_nodes))
        probabilities[nodes] = probabilities.get(nodes, 0.0) + float(weight)
    ordered = sorted(probabilities.items(), key=lambda item: _node_label(item[0]))
    credible = sorted(
        probabilities.items(),
        key=lambda item: (-item[1], _node_label(item[0])),
    )
    selected: list[tuple[int, ...]] = []
    cumulative = 0.0
    for nodes, probability in credible:
        selected.append(nodes)
        cumulative += probability
        if cumulative >= confidence_level:
            break
    positive = np.asarray(
        [probability for _, probability in ordered if probability > 0.0],
        dtype=float,
    )
    entropy = -float(np.sum(positive * np.log(positive)))
    support_size = len(ordered)
    return {
        "node_posterior_entropy": entropy,
        "node_posterior_normalized_entropy": (
            entropy / np.log(support_size) if support_size > 1 else 0.0
        ),
        "node_posterior_effective_sample_size": float(
            1.0 / np.sum(np.square(positive))
        ),
        "node_posterior_support_size": support_size,
        "node_credible_set_size": len(selected),
        "node_posterior": json.dumps(
            {
                _node_label(nodes): probability
                for nodes, probability in ordered
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "node_credible_set": json.dumps(
            [_node_label(nodes) for nodes in selected],
            separators=(",", ":"),
        ),
    }


def _recompute_online_posterior_rows(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    benchmark_config = CounterfactualBenchmarkConfig(
        **dict(summary["benchmark_config"])
    )
    contact_config = LatentContactConfig(**dict(summary["contact_config"]))
    seeds = [int(seed) for seed in summary["seeds"]]
    protocols = build_protocol(benchmark_config)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        fitted: list[_FittedObject] = []
        for object_index, protocol in enumerate(protocols):
            training, validation, held_out = generate_episodes(
                protocol,
                benchmark_config,
                seed=seed * 10_000 + object_index * 101,
            )
            baselines = fit_baselines(
                training,
                validation,
                make_parameter_grid(protocol.graph_object, benchmark_config),
                benchmark_config,
            )
            fitted.append(
                _FittedObject(
                    protocol=protocol,
                    training=training,
                    validation=validation,
                    held_out=held_out,
                    baselines=baselines,
                )
            )

        for target_index, target in enumerate(fitted):
            sources = tuple(
                item for index, item in enumerate(fitted) if index != target_index
            )
            source_protocols = tuple(item.protocol for item in sources)
            prior = fit_contact_prior(
                source_protocols,
                contact_config,
                action_split="test",
            )
            model = GraphContactHypothesisModel(
                prior=prior,
                config=contact_config,
            )
            calibration = _calibrate_fold(
                sources,
                model,
                benchmark_config,
                contact_config,
                calibration_seed=(
                    seed * 1_000_003 + target_index * 100_003 + 17
                ),
            )
            bank = build_rollout_bank(
                target.protocol.graph_object,
                target.protocol.test_action,
                target.baselines.physics.posterior,
                model,
                simulator_config=benchmark_config.simulator,
                parameter_particle_count=contact_config.parameter_particle_count,
                variance_floor_m2=(
                    benchmark_config.predictive_variance_floor_m2
                ),
                confidence_level=contact_config.confidence_level,
            )
            prefix = contact_config.prefix_frame_count(
                benchmark_config.frame_count
            )
            source_names = tuple(
                source.protocol.graph_object.name for source in sources
            )
            for condition_index, episode in enumerate(target.held_out):
                observation_rng = np.random.default_rng(
                    seed * 1_000_003
                    + target_index * 10_007
                    + condition_index * 97
                )
                observations = episode.truth + observation_rng.normal(
                    scale=contact_config.observation_noise_std_m,
                    size=episode.truth.shape,
                )
                raw_weights = bank.update_weights(
                    observations,
                    prefix_frame_count=prefix,
                    likelihood_scale_m=calibration.likelihood_scale_m,
                    likelihood_power=calibration.likelihood_power,
                    dynamic_likelihood_weight=(
                        calibration.dynamic_likelihood_weight
                    ),
                )
                joint_weights = _temper_joint_weights(
                    raw_weights,
                    calibration.posterior_temperature,
                )
                contact_weights = bank.contact_marginal(joint_weights)
                truth = true_contact_state(
                    target.protocol.graph_object,
                    episode.action,
                    episode.condition,
                )
                rows.append(
                    {
                        "seed": seed,
                        "object": target.protocol.graph_object.name,
                        "source_objects": ";".join(source_names),
                        "world_condition": episode.condition.name,
                        "setting": "online_adaptation",
                        "observation_fraction": (
                            contact_config.observation_fraction
                        ),
                        **contact_recovery_metrics(
                            bank.contact_states,
                            contact_weights,
                            truth,
                            confidence_level=contact_config.confidence_level,
                        ),
                        **_posterior_diagnostic_metrics(
                            bank.contact_states,
                            contact_weights,
                            confidence_level=contact_config.confidence_level,
                        ),
                    }
                )
    return rows


def _row_key(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(row["seed"]),
        str(row["object"]),
        str(row["world_condition"]),
        str(row["setting"]),
    )


def _validate_recomputed_parity(
    retained_rows: Sequence[Mapping[str, Any]],
    recomputed_rows: Sequence[Mapping[str, Any]],
    config: DiagnosticConfig,
) -> dict[str, Any]:
    retained = {
        _row_key(row): row
        for row in retained_rows
        if str(row["setting"]) == "online_adaptation"
    }
    recomputed = {_row_key(row): row for row in recomputed_rows}
    if set(retained) != set(recomputed):
        missing = sorted(set(retained) - set(recomputed))
        extra = sorted(set(recomputed) - set(retained))
        raise ValueError(
            "recomputed posterior keys differ from retained bundle; "
            f"missing={missing!r}, extra={extra!r}"
        )
    numeric_checks = 0
    exact_checks = 0
    maximum_absolute_difference = 0.0
    for key in sorted(retained):
        expected = retained[key]
        actual = recomputed[key]
        for field_name in _EXACT_PARITY_FIELDS:
            expected_value: Any = expected[field_name]
            actual_value: Any = actual[field_name]
            if field_name in {
                "node_correct",
                "node_credible_covered",
                "delay_map_correct",
            }:
                expected_value = _as_bool(expected_value)
                actual_value = _as_bool(actual_value)
            elif field_name == "delay_map":
                expected_value = int(expected_value)
                actual_value = int(actual_value)
            if expected_value != actual_value:
                raise ValueError(
                    f"{key}/{field_name}: retained {expected_value!r}, "
                    f"recomputed {actual_value!r}"
                )
            exact_checks += 1
        for field_name in _FLOAT_PARITY_FIELDS:
            expected_value = float(expected[field_name])
            actual_value = float(actual[field_name])
            difference = abs(expected_value - actual_value)
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                difference,
            )
            if not np.isclose(
                expected_value,
                actual_value,
                rtol=config.parity_relative_tolerance,
                atol=config.parity_absolute_tolerance,
            ):
                raise ValueError(
                    f"{key}/{field_name}: retained {expected_value!r}, "
                    f"recomputed {actual_value!r}"
                )
            numeric_checks += 1
    return {
        "passed": True,
        "row_count": len(retained),
        "exact_checks": exact_checks,
        "numeric_checks": numeric_checks,
        "maximum_absolute_difference": maximum_absolute_difference,
    }


def _trajectory_lookup(
    intervention_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str, str], dict[str, float]]:
    grouped: dict[tuple[int, str, str, str], dict[str, float]] = {}
    for row in intervention_rows:
        if str(row["setting"]) != "online_adaptation":
            continue
        key = _row_key(row)
        grouped.setdefault(key, {})[str(row["method"])] = float(
            row["trajectory_rmse_m"]
        )
    return grouped


def _action_direction(graph: GraphObject, frame_count: int) -> np.ndarray:
    test_action = next(
        action
        for action in make_actions(graph, frame_count)
        if action.split == "test"
    )
    vector = np.sum(test_action.commanded_forces, axis=(0, 1))
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("test action has no net commanded direction")
    return vector / norm


def _enrich_rows(
    posterior_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, GraphObject],
    *,
    frame_count: int,
    config: DiagnosticConfig,
) -> list[dict[str, Any]]:
    trajectories = _trajectory_lookup(intervention_rows)
    output: list[dict[str, Any]] = []
    for source_row in posterior_rows:
        if str(source_row["world_condition"]) != "shifted_contact":
            continue
        row = dict(source_row)
        key = _row_key(row)
        methods = trajectories.get(key, {})
        if "nominal_physics" not in methods or "latent_contact" not in methods:
            raise ValueError(f"missing paired trajectory methods for {key}")
        nominal_rmse = methods["nominal_physics"]
        latent_rmse = methods["latent_contact"]
        trajectory_gain = nominal_rmse - latent_rmse
        graph = graphs[str(row["object"])]
        distances = _all_pairs_graph_distances(graph)
        truth = _parse_nodes(str(row["node_truth"]))
        estimate = _parse_nodes(str(row["node_map"]))
        mean_distance, maximum_distance = _assignment_distances(
            truth,
            estimate,
            distances,
        )
        truth_field = _diffused_force_field(
            graph,
            truth,
            strength=config.diffusion_strength,
        )
        estimate_field = _diffused_force_field(
            graph,
            estimate,
            strength=config.diffusion_strength,
        )
        force_similarity = float(np.dot(truth_field, estimate_field))
        graph_symmetry = _assignment_signature(
            graph,
            truth,
            distances,
            include_sensors=False,
        ) == _assignment_signature(
            graph,
            estimate,
            distances,
            include_sensors=False,
        )
        sensor_symmetry = _assignment_signature(
            graph,
            truth,
            distances,
            include_sensors=True,
        ) == _assignment_signature(
            graph,
            estimate,
            distances,
            include_sensors=True,
        )
        adjacency = _adjacency(graph)
        truth_degree = float(np.mean([len(adjacency[node]) for node in truth]))
        estimate_degree = float(
            np.mean([len(adjacency[node]) for node in estimate])
        )
        truth_sensor_distance = float(
            np.mean(
                [
                    min(int(distances[node, sensor]) for sensor in graph.sensor_nodes)
                    for node in truth
                ]
            )
        )
        estimate_sensor_distance = float(
            np.mean(
                [
                    min(int(distances[node, sensor]) for sensor in graph.sensor_nodes)
                    for node in estimate
                ]
            )
        )
        truth_position = np.mean(graph.rest_positions[list(truth)], axis=0)
        estimate_position = np.mean(graph.rest_positions[list(estimate)], axis=0)
        direction = _action_direction(graph, frame_count)
        action_projection_difference = abs(
            float(np.dot(estimate_position - truth_position, direction))
        )
        exact = truth == estimate
        one_hop = maximum_distance <= 1
        force_equivalent = (
            force_similarity >= config.force_field_equivalence_threshold
        )
        if exact:
            category = "exact"
        elif sensor_symmetry and trajectory_gain > 0.0:
            category = "symmetry_metric_limitation"
        elif one_hop and force_equivalent and trajectory_gain > 0.0:
            category = "trajectory_equivalent_neighbor"
        else:
            category = "genuinely_wrong_or_unresolved"
        row.update(
            {
                "exact_node_recovered": exact,
                "one_hop_patch_recovered": one_hop,
                "mean_assignment_graph_distance": mean_distance,
                "maximum_assignment_graph_distance": maximum_distance,
                "graph_symmetry_proxy": graph_symmetry,
                "sensor_conditioned_symmetry_proxy": sensor_symmetry,
                "truth_mean_degree": truth_degree,
                "map_mean_degree": estimate_degree,
                "truth_mean_nearest_sensor_distance": truth_sensor_distance,
                "map_mean_nearest_sensor_distance": estimate_sensor_distance,
                "rest_position_error_m": float(
                    np.linalg.norm(estimate_position - truth_position)
                ),
                "action_projection_error_m": action_projection_difference,
                "force_field_proxy_cosine": force_similarity,
                "force_field_proxy_equivalent": force_equivalent,
                "nominal_trajectory_rmse_m": nominal_rmse,
                "latent_trajectory_rmse_m": latent_rmse,
                "trajectory_rmse_gain_m": trajectory_gain,
                "trajectory_relative_gain": (
                    trajectory_gain / max(nominal_rmse, 1e-15)
                ),
                "trajectory_improved": trajectory_gain > 0.0,
                "diagnostic_category": category,
            }
        )
        for threshold in _FORCE_FIELD_THRESHOLDS:
            label = str(threshold).replace(".", "_")
            row[f"force_field_proxy_recovered_at_{label}"] = (
                force_similarity >= threshold
            )
        output.append(row)
    return output


def _mean(rows: Sequence[Mapping[str, Any]], field_name: str) -> float:
    return float(np.mean([float(row[field_name]) for row in rows]))


def _fraction(rows: Sequence[Mapping[str, Any]], field_name: str) -> float:
    return float(np.mean([float(_as_bool(row[field_name])) for row in rows]))


def _confusion_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    labels = sorted(
        {
            str(row[field_name])
            for row in rows
            for field_name in ("node_truth", "node_map")
        }
    )
    matrix = {truth: {estimate: 0 for estimate in labels} for truth in labels}
    for row in rows:
        matrix[str(row["node_truth"])][str(row["node_map"])] += 1
    return matrix


def _trajectory_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    exact: bool,
) -> dict[str, Any]:
    selected = [
        row for row in rows if _as_bool(row["exact_node_recovered"]) is exact
    ]
    if not selected:
        return {
            "case_count": 0,
            "mean_gain_m": None,
            "mean_relative_gain": None,
            "improved_fraction": None,
        }
    return {
        "case_count": len(selected),
        "mean_gain_m": _mean(selected, "trajectory_rmse_gain_m"),
        "mean_relative_gain": _mean(selected, "trajectory_relative_gain"),
        "improved_fraction": _fraction(selected, "trajectory_improved"),
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty diagnostic rows")
    exact_accuracy = _fraction(rows, "exact_node_recovered")
    mean_confidence = _mean(rows, "node_confidence")
    result: dict[str, Any] = {
        "case_count": len(rows),
        "exact_node_accuracy": exact_accuracy,
        "one_hop_patch_accuracy": _fraction(
            rows,
            "one_hop_patch_recovered",
        ),
        "mean_assignment_graph_distance": _mean(
            rows,
            "mean_assignment_graph_distance",
        ),
        "mean_maximum_assignment_graph_distance": _mean(
            rows,
            "maximum_assignment_graph_distance",
        ),
        "mean_node_confidence": mean_confidence,
        "node_confidence_calibration_error": abs(
            mean_confidence - exact_accuracy
        ),
        "mean_node_truth_probability": _mean(
            rows,
            "node_truth_probability",
        ),
        "mean_node_brier": _mean(rows, "node_brier"),
        "node_credible_coverage": _fraction(
            rows,
            "node_credible_covered",
        ),
        "mean_node_credible_set_size": _mean(
            rows,
            "node_credible_set_size",
        ),
        "mean_node_posterior_entropy": _mean(
            rows,
            "node_posterior_entropy",
        ),
        "mean_node_posterior_normalized_entropy": _mean(
            rows,
            "node_posterior_normalized_entropy",
        ),
        "mean_node_posterior_effective_sample_size": _mean(
            rows,
            "node_posterior_effective_sample_size",
        ),
        "mean_joint_normalized_entropy": _mean(
            rows,
            "joint_normalized_entropy",
        ),
        "mean_joint_effective_sample_size": _mean(
            rows,
            "joint_effective_sample_size",
        ),
        "mean_force_field_proxy_cosine": _mean(
            rows,
            "force_field_proxy_cosine",
        ),
        "graph_symmetry_proxy_fraction": _fraction(
            rows,
            "graph_symmetry_proxy",
        ),
        "sensor_conditioned_symmetry_proxy_fraction": _fraction(
            rows,
            "sensor_conditioned_symmetry_proxy",
        ),
        "mean_truth_degree": _mean(rows, "truth_mean_degree"),
        "mean_truth_nearest_sensor_distance": _mean(
            rows,
            "truth_mean_nearest_sensor_distance",
        ),
        "mean_action_projection_error_m": _mean(
            rows,
            "action_projection_error_m",
        ),
        "trajectory_exact_map": _trajectory_subset(rows, exact=True),
        "trajectory_incorrect_map": _trajectory_subset(rows, exact=False),
        "diagnostic_category_counts": {
            category: sum(
                str(row["diagnostic_category"]) == category for row in rows
            )
            for category in sorted(
                {str(row["diagnostic_category"]) for row in rows}
            )
        },
        "confusion_matrix": _confusion_matrix(rows),
    }
    for threshold in _FORCE_FIELD_THRESHOLDS:
        label = str(threshold).replace(".", "_")
        result[f"force_field_proxy_recovery_at_{label}"] = _fraction(
            rows,
            f"force_field_proxy_recovered_at_{label}",
        )
    return result


def _grouped_aggregates(
    rows: Sequence[Mapping[str, Any]],
    field_name: str,
) -> list[dict[str, Any]]:
    values = sorted({str(row[field_name]) for row in rows})
    return [
        {
            field_name: value,
            **_aggregate_rows(
                [row for row in rows if str(row[field_name]) == value]
            ),
        }
        for value in values
    ]


def _degree_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (str(row["object"]), float(row["truth_mean_degree"]))
            for row in rows
        }
    )
    return [
        {
            "object": object_name,
            "truth_mean_degree": degree,
            **_aggregate_rows(
                [
                    row
                    for row in rows
                    if str(row["object"]) == object_name
                    and float(row["truth_mean_degree"]) == degree
                ]
            ),
        }
        for object_name, degree in keys
    ]


def _sensor_distance_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (
                str(row["object"]),
                float(row["truth_mean_nearest_sensor_distance"]),
            )
            for row in rows
        }
    )
    return [
        {
            "object": object_name,
            "truth_mean_nearest_sensor_distance": distance,
            **_aggregate_rows(
                [
                    row
                    for row in rows
                    if str(row["object"]) == object_name
                    and float(row["truth_mean_nearest_sensor_distance"])
                    == distance
                ]
            ),
        }
        for object_name, distance in keys
    ]


def analyze_contact_posterior_bundle(
    bundle_directory: str | Path,
    *,
    config: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Recompute and diagnose a controlled latent-contact result bundle."""

    diagnostic_config = config or DiagnosticConfig()
    bundle = Path(bundle_directory).resolve()
    summary_path = bundle / "summary.json"
    manifest_path = bundle / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "bundle must contain summary.json and manifest.json"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("benchmark") != "causal4d-latent-contact-v1":
        raise ValueError("unsupported benchmark bundle")
    retained_rows = _read_csv(bundle / "contact_recovery.csv")
    intervention_rows = _read_csv(bundle / "interventions.csv")
    recomputed_rows = _recompute_online_posterior_rows(summary)
    parity = _validate_recomputed_parity(
        retained_rows,
        recomputed_rows,
        diagnostic_config,
    )
    benchmark_config = CounterfactualBenchmarkConfig(
        **dict(summary["benchmark_config"])
    )
    graphs = {
        protocol.graph_object.name: protocol.graph_object
        for protocol in build_protocol(benchmark_config)
    }
    rows = _enrich_rows(
        recomputed_rows,
        intervention_rows,
        graphs,
        frame_count=benchmark_config.frame_count,
        config=diagnostic_config,
    )
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactPosteriorDiagnostic",
        "source_bundle": {
            "directory": str(bundle),
            "manifest_sha256": _sha256(manifest_path),
            "benchmark": summary["benchmark"],
            "seeds": summary["seeds"],
        },
        "diagnostic_config": diagnostic_config.as_dict(),
        "recomputation_parity": parity,
        "overall": _aggregate_rows(rows),
        "by_topology": _grouped_aggregates(rows, "object"),
        "by_seed": _grouped_aggregates(rows, "seed"),
        "by_truth_degree": _degree_diagnostics(rows),
        "by_truth_sensor_distance": _sensor_distance_diagnostics(rows),
        "rows": rows,
        "registered_success_gates": summary["success_gates"],
        "registered_gate_status_unchanged": True,
        "claim_boundary": (
            "Controlled diagnostic only. Exact-node gates, thresholds, the "
            "frozen estimator, and the registered real experiment are unchanged. "
            "The force-field measure is a declared graph-diffusion proxy, not a "
            "replacement contact label or a physical-equivalence claim."
        ),
    }


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    overall = result["overall"]
    lines = [
        "# Contact-posterior diagnostic",
        "",
        "This is a controlled diagnostic. It does not modify the frozen estimator,",
        "registered exact-node gate, thresholds, or real experiment.",
        "",
        "## Overall shifted-contact result",
        "",
        f"- Cases: `{overall['case_count']}`",
        f"- Exact-node accuracy: `{overall['exact_node_accuracy']:.3%}`",
        f"- One-hop-patch accuracy: `{overall['one_hop_patch_accuracy']:.3%}`",
        (
            "- Mean exact-node confidence calibration error: "
            f"`{overall['node_confidence_calibration_error']:.3%}`"
        ),
        (
            "- Mean graph-diffusion force-field cosine: "
            f"`{overall['mean_force_field_proxy_cosine']:.4f}`"
        ),
        "",
        "## By topology",
        "",
        (
            "| Topology | Cases | Exact | One hop | Confidence | "
            "Calibration error | Mean graph distance |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["by_topology"]:
        lines.append(
            "| {object} | {case_count} | {exact:.1%} | {one_hop:.1%} | "
            "{confidence:.1%} | {calibration:.1%} | {distance:.3f} |".format(
                object=row["object"],
                case_count=row["case_count"],
                exact=row["exact_node_accuracy"],
                one_hop=row["one_hop_patch_accuracy"],
                confidence=row["mean_node_confidence"],
                calibration=row["node_confidence_calibration_error"],
                distance=row["mean_assignment_graph_distance"],
            )
        )
    lines.extend(
        [
            "",
            "## Trajectory effect conditional on exact-node MAP",
            "",
            (
                "- Correct MAP subset: "
                f"`{overall['trajectory_exact_map']['case_count']}` cases."
            ),
            (
                "- Incorrect MAP subset: "
                f"`{overall['trajectory_incorrect_map']['case_count']}` cases; "
                "trajectory gain is reported separately in the JSON artifact."
            ),
            "",
            "Diagnostic categories distinguish exact recovery, a sensor-conditioned",
            "symmetry proxy, a one-hop graph-diffusion-equivalent neighbor, and",
            "genuinely wrong or unresolved posteriors. The proxy categories do not",
            "replace the registered exact-node metric.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_contact_posterior_diagnostics(
    result: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Write immutable JSON, CSV, Markdown, and manifest diagnostics."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "contact-posterior-diagnostics.json"
    rows_path = output / "contact-posterior-rows.csv"
    markdown_path = output / "contact-posterior-diagnostics.md"
    _write_json(
        summary_path,
        {key: value for key, value in result.items() if key != "rows"},
    )
    _write_csv(rows_path, list(result["rows"]))
    _write_markdown(markdown_path, result)
    payloads = (summary_path, rows_path, markdown_path)
    manifest_path = output / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifact_kind": "Causal4DContactPosteriorDiagnosticManifest",
            "artifacts": {
                path.name: {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in payloads
            },
        },
    )
    return {
        "summary": str(summary_path),
        "rows": str(rows_path),
        "markdown": str(markdown_path),
        "manifest": str(manifest_path),
    }
