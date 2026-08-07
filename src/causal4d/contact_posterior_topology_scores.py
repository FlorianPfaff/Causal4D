"""Full-posterior topology scores for latent contact diagnostics.

The scores in this module are analysis-only. They do not modify the frozen
latent-contact estimator, any registered success gate, or the physical
experiment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import permutations
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.benchmark import CounterfactualBenchmarkConfig, build_protocol
from causal4d.simulator import GraphObject


_DEFAULT_FORCE_FIELD_THRESHOLDS = (0.80, 0.90, 0.95)


@dataclass(frozen=True)
class TopologyPosteriorScoreConfig:
    """Declared analysis choices for full-posterior topology scoring."""

    confidence_level: float = 0.90
    diffusion_strength: float = 1.0
    force_field_similarity_thresholds: tuple[float, ...] = (
        _DEFAULT_FORCE_FIELD_THRESHOLDS
    )

    def __post_init__(self) -> None:
        if not np.isfinite(self.confidence_level) or not (
            0.0 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must be finite and in (0, 1)")
        if not np.isfinite(self.diffusion_strength) or self.diffusion_strength <= 0.0:
            raise ValueError("diffusion_strength must be finite and positive")
        thresholds = tuple(self.force_field_similarity_thresholds)
        if not thresholds:
            raise ValueError("at least one force-field threshold is required")
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds
        ):
            raise ValueError("force-field thresholds must be finite and in [0, 1]")
        if tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError("force-field thresholds must be unique and sorted")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _threshold_label(value: float) -> str:
    return str(value).replace(".", "_")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def parse_contact_assignment_label(value: str) -> tuple[int, ...]:
    """Parse the canonical semicolon-separated contact-assignment label."""

    _require(isinstance(value, str) and bool(value.strip()), "contact label is empty")
    try:
        nodes = tuple(int(part) for part in value.split(";"))
    except ValueError as error:
        raise ValueError("contact label contains a non-integer node") from error
    _require(bool(nodes), "contact assignment must be nonempty")
    _require(len(nodes) == len(set(nodes)), "contact assignment contains duplicates")
    _require(all(node >= 0 for node in nodes), "contact nodes must be nonnegative")
    return nodes


def parse_node_posterior(value: str) -> dict[tuple[int, ...], float]:
    """Parse and validate the serialized categorical contact posterior."""

    _require(isinstance(value, str) and bool(value), "node posterior is missing")
    try:
        payload = json.loads(value, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ValueError("node posterior is invalid JSON") from error
    _require(isinstance(payload, Mapping) and bool(payload), "node posterior is empty")
    probabilities: dict[tuple[int, ...], float] = {}
    for label, raw_probability in payload.items():
        nodes = parse_contact_assignment_label(str(label))
        _require(nodes not in probabilities, "node posterior contains duplicate labels")
        _require(
            isinstance(raw_probability, (int, float))
            and not isinstance(raw_probability, bool),
            "node posterior probabilities must be numeric",
        )
        probability = float(raw_probability)
        _require(
            np.isfinite(probability) and probability >= 0.0,
            "node posterior probabilities must be finite and nonnegative",
        )
        probabilities[nodes] = probability
    total = float(sum(probabilities.values()))
    _require(
        np.isclose(total, 1.0, rtol=1e-10, atol=1e-12),
        "node posterior probabilities must sum to one",
    )
    if total != 1.0:
        probabilities = {
            assignment: probability / total
            for assignment, probability in probabilities.items()
        }
    return probabilities


def parse_contact_credible_set(value: str) -> tuple[tuple[int, ...], ...]:
    """Parse the serialized ordered node credible set."""

    _require(isinstance(value, str) and bool(value), "node credible set is missing")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("node credible set is invalid JSON") from error
    _require(isinstance(payload, list) and bool(payload), "node credible set is empty")
    assignments = tuple(parse_contact_assignment_label(str(item)) for item in payload)
    _require(
        len(assignments) == len(set(assignments)),
        "node credible set contains duplicate assignments",
    )
    return assignments


def all_pairs_graph_distances(graph: GraphObject) -> np.ndarray:
    """Return connected, unweighted shortest-path distances between all nodes."""

    count = int(graph.rest_positions.shape[0])
    _require(count > 0, "graph must contain at least one node")
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for first, second in graph.edges:
        first = int(first)
        second = int(second)
        _require(
            0 <= first < count and 0 <= second < count and first != second,
            "graph contains an invalid edge",
        )
        adjacency[first].append(second)
        adjacency[second].append(first)

    distances = np.full((count, count), count + 1, dtype=int)
    for source in range(count):
        distances[source, source] = 0
        frontier = [source]
        cursor = 0
        while cursor < len(frontier):
            node = frontier[cursor]
            cursor += 1
            candidate_distance = int(distances[source, node]) + 1
            for neighbor in adjacency[node]:
                if candidate_distance < distances[source, neighbor]:
                    distances[source, neighbor] = candidate_distance
                    frontier.append(neighbor)
    _require(not np.any(distances > count), f"graph {graph.name!r} is disconnected")
    return distances


def assignment_graph_distance(
    truth: Sequence[int],
    estimate: Sequence[int],
    distances: np.ndarray,
) -> tuple[float, int]:
    """Return minimum mean and maximum hop distance over contact permutations."""

    truth_nodes = tuple(map(int, truth))
    estimate_nodes = tuple(map(int, estimate))
    _require(bool(truth_nodes), "contact assignment must be nonempty")
    _require(
        len(truth_nodes) == len(estimate_nodes),
        "contact assignments have different cardinalities",
    )
    _require(
        len(truth_nodes) == len(set(truth_nodes))
        and len(estimate_nodes) == len(set(estimate_nodes)),
        "contact assignments contain duplicate nodes",
    )
    matrix = np.asarray(distances)
    _require(
        matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1],
        "graph-distance matrix must be square",
    )
    count = matrix.shape[0]
    _require(
        all(0 <= node < count for node in (*truth_nodes, *estimate_nodes)),
        "contact assignment references an invalid node",
    )

    candidates: list[tuple[int, int]] = []
    for assignment in permutations(estimate_nodes):
        values = [
            int(matrix[truth_node, estimate_node])
            for truth_node, estimate_node in zip(truth_nodes, assignment, strict=True)
        ]
        candidates.append((sum(values), max(values, default=0)))
    total, maximum = min(candidates)
    return float(total / len(truth_nodes)), int(maximum)


def diffused_contact_field(
    graph: GraphObject,
    nodes: Sequence[int],
    *,
    diffusion_strength: float,
) -> np.ndarray:
    """Return a unit graph-diffusion proxy for one contact assignment."""

    _require(
        np.isfinite(diffusion_strength) and diffusion_strength > 0.0,
        "diffusion_strength must be finite and positive",
    )
    assignment = tuple(map(int, nodes))
    count = int(graph.rest_positions.shape[0])
    _require(bool(assignment), "contact assignment must be nonempty")
    _require(
        len(assignment) == len(set(assignment)),
        "contact assignment contains duplicate nodes",
    )
    _require(
        all(0 <= node < count for node in assignment),
        "contact assignment references an invalid node",
    )
    adjacency = np.zeros((count, count), dtype=float)
    for first, second in graph.edges:
        adjacency[int(first), int(second)] = 1.0
        adjacency[int(second), int(first)] = 1.0
    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    source = np.zeros(count, dtype=float)
    for node in assignment:
        source[node] += 1.0 / len(assignment)
    field = np.linalg.solve(
        np.eye(count, dtype=float) + diffusion_strength * laplacian,
        source,
    )
    norm = float(np.linalg.norm(field))
    _require(
        np.isfinite(norm) and norm > 0.0,
        "graph diffusion produced an invalid force-field proxy",
    )
    return field / norm


def _credible_set_distances(
    credible_set: Sequence[Sequence[int]],
    truth: tuple[int, ...],
    distances: np.ndarray,
) -> dict[str, Any]:
    values = [
        assignment_graph_distance(truth, assignment, distances)
        for assignment in credible_set
    ]
    minimum_mean = min(value[0] for value in values)
    minimum_max = min(value[1] for value in values)
    return {
        "node_credible_set_min_mean_graph_distance_hops": minimum_mean,
        "node_credible_set_min_max_graph_distance_hops": minimum_max,
        "node_credible_set_one_hop_patch_covered": minimum_max <= 1,
    }


def posterior_topology_scores(
    graph: GraphObject,
    probabilities: Mapping[tuple[int, ...], float],
    truth: Sequence[int],
    *,
    config: TopologyPosteriorScoreConfig,
    credible_set: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Score the complete node posterior in graph and force-field space."""

    truth_nodes = tuple(map(int, truth))
    _require(bool(probabilities), "node posterior must be nonempty")
    normalized: dict[tuple[int, ...], float] = {}
    for raw_assignment, raw_probability in probabilities.items():
        assignment = tuple(map(int, raw_assignment))
        _require(
            len(assignment) == len(truth_nodes),
            "posterior contact cardinality differs from truth",
        )
        _require(
            assignment not in normalized,
            "node posterior contains duplicate assignments",
        )
        probability = float(raw_probability)
        _require(
            np.isfinite(probability) and probability >= 0.0,
            "node posterior probabilities must be finite and nonnegative",
        )
        normalized[assignment] = probability
    total = float(sum(normalized.values()))
    _require(
        np.isclose(total, 1.0, rtol=1e-10, atol=1e-12),
        "node posterior probabilities must sum to one",
    )
    if total != 1.0:
        normalized = {
            assignment: probability / total
            for assignment, probability in normalized.items()
        }

    support = tuple(sorted(normalized))
    weights = np.asarray([normalized[assignment] for assignment in support])
    distances = all_pairs_graph_distances(graph)
    graph_distances = [
        assignment_graph_distance(truth_nodes, assignment, distances)
        for assignment in support
    ]
    mean_distances = np.asarray([value[0] for value in graph_distances])
    maximum_distances = np.asarray([value[1] for value in graph_distances], dtype=int)

    truth_field = diffused_contact_field(
        graph,
        truth_nodes,
        diffusion_strength=config.diffusion_strength,
    )
    fields = np.stack(
        [
            diffused_contact_field(
                graph,
                assignment,
                diffusion_strength=config.diffusion_strength,
            )
            for assignment in support
        ]
    )
    cosines = np.clip(fields @ truth_field, -1.0, 1.0)
    field_distances = np.linalg.norm(fields - truth_field[None, :], axis=1)
    pairwise_distances = np.linalg.norm(
        fields[:, None, :] - fields[None, :, :],
        axis=2,
    )
    expected_field_distance = float(weights @ field_distances)
    pairwise_dispersion = 0.5 * float(
        np.sum(weights[:, None] * weights[None, :] * pairwise_distances)
    )
    energy_score = expected_field_distance - pairwise_dispersion
    if energy_score < 0.0 and np.isclose(energy_score, 0.0, atol=1e-14):
        energy_score = 0.0
    _require(energy_score >= 0.0, "force-field energy score became negative")

    mass_by_radius: dict[int, float] = {}
    for radius in sorted(set(map(int, maximum_distances))):
        mass_by_radius[radius] = float(np.sum(weights[maximum_distances <= radius]))
    credible_radius = next(
        radius
        for radius, mass in sorted(mass_by_radius.items())
        if mass >= config.confidence_level - 1e-12
    )

    result: dict[str, Any] = {
        "posterior_expected_assignment_graph_distance_hops": float(
            weights @ mean_distances
        ),
        "posterior_expected_max_assignment_graph_distance_hops": float(
            weights @ maximum_distances
        ),
        "posterior_exact_node_mass": float(normalized.get(truth_nodes, 0.0)),
        "posterior_one_hop_patch_mass": float(np.sum(weights[maximum_distances <= 1])),
        "posterior_two_hop_patch_mass": float(np.sum(weights[maximum_distances <= 2])),
        "posterior_truth_centered_credible_radius_hops": int(credible_radius),
        "posterior_expected_force_field_cosine": float(weights @ cosines),
        "posterior_expected_force_field_distance": expected_field_distance,
        "posterior_force_field_pairwise_dispersion": pairwise_dispersion,
        "posterior_force_field_energy_score": float(energy_score),
    }
    for threshold in config.force_field_similarity_thresholds:
        result[f"posterior_force_field_mass_at_{_threshold_label(threshold)}"] = float(
            np.sum(weights[cosines >= threshold])
        )
    if credible_set is not None:
        result.update(
            _credible_set_distances(
                credible_set,
                truth_nodes,
                distances,
            )
        )
    return result


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _fraction(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(bool(row[field])) for row in rows]))


def aggregate_posterior_topology_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: TopologyPosteriorScoreConfig,
) -> dict[str, Any]:
    """Aggregate the additive full-posterior topology diagnostics."""

    _require(bool(rows), "cannot aggregate empty topology-score rows")
    result: dict[str, Any] = {
        "case_count": len(rows),
        "mean_posterior_expected_assignment_graph_distance_hops": _mean(
            rows,
            "posterior_expected_assignment_graph_distance_hops",
        ),
        "mean_posterior_expected_max_assignment_graph_distance_hops": _mean(
            rows,
            "posterior_expected_max_assignment_graph_distance_hops",
        ),
        "mean_posterior_exact_node_mass": _mean(rows, "posterior_exact_node_mass"),
        "mean_posterior_one_hop_patch_mass": _mean(
            rows,
            "posterior_one_hop_patch_mass",
        ),
        "mean_posterior_two_hop_patch_mass": _mean(
            rows,
            "posterior_two_hop_patch_mass",
        ),
        "mean_truth_centered_credible_radius_hops": _mean(
            rows,
            "posterior_truth_centered_credible_radius_hops",
        ),
        "mean_posterior_expected_force_field_cosine": _mean(
            rows,
            "posterior_expected_force_field_cosine",
        ),
        "mean_posterior_expected_force_field_distance": _mean(
            rows,
            "posterior_expected_force_field_distance",
        ),
        "mean_posterior_force_field_pairwise_dispersion": _mean(
            rows,
            "posterior_force_field_pairwise_dispersion",
        ),
        "mean_posterior_force_field_energy_score": _mean(
            rows,
            "posterior_force_field_energy_score",
        ),
    }
    if all("node_credible_set_one_hop_patch_covered" in row for row in rows):
        result["node_credible_set_one_hop_patch_coverage"] = _fraction(
            rows,
            "node_credible_set_one_hop_patch_covered",
        )
        result["mean_node_credible_set_min_graph_distance_hops"] = _mean(
            rows,
            "node_credible_set_min_mean_graph_distance_hops",
        )
    for threshold in config.force_field_similarity_thresholds:
        field = f"posterior_force_field_mass_at_{_threshold_label(threshold)}"
        result[f"mean_{field}"] = _mean(rows, field)
    return result


def augment_contact_posterior_result(
    result: Mapping[str, Any],
    *,
    graphs: Mapping[str, GraphObject],
    config: TopologyPosteriorScoreConfig,
) -> dict[str, Any]:
    """Add full-posterior topology scores to one admitted diagnostic result."""

    output = deepcopy(dict(result))
    source_rows = output.get("rows")
    _require(
        isinstance(source_rows, list) and bool(source_rows),
        "diagnostic rows missing",
    )
    rows: list[dict[str, Any]] = []
    for raw_row in source_rows:
        _require(isinstance(raw_row, Mapping), "diagnostic row must be an object")
        row = dict(raw_row)
        object_name = str(row.get("object"))
        _require(object_name in graphs, f"unknown diagnostic topology: {object_name}")
        probabilities = parse_node_posterior(str(row.get("node_posterior", "")))
        truth = parse_contact_assignment_label(str(row.get("node_truth", "")))
        credible_set = parse_contact_credible_set(str(row.get("node_credible_set", "")))
        row.update(
            posterior_topology_scores(
                graphs[object_name],
                probabilities,
                truth,
                config=config,
                credible_set=credible_set,
            )
        )
        rows.append(row)

    overall = aggregate_posterior_topology_scores(rows, config=config)
    by_topology = [
        {
            "object": object_name,
            **aggregate_posterior_topology_scores(
                [row for row in rows if str(row["object"]) == object_name],
                config=config,
            ),
        }
        for object_name in sorted({str(row["object"]) for row in rows})
    ]
    output["rows"] = rows
    output["posterior_topology_scores"] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactPosteriorTopologyScores",
        "config": config.as_dict(),
        "overall": overall,
        "by_topology": by_topology,
        "registered_success_gates_unchanged": True,
        "claim_boundary": (
            "Additive controlled diagnostic only. Exact-node gates, thresholds, "
            "the frozen estimator, and the registered physical experiment are "
            "unchanged. The force-field energy score is proper only for the "
            "declared graph-diffusion proxy representation."
        ),
    }
    output["registered_gate_status_unchanged"] = True
    original_boundary = str(output.get("claim_boundary", "")).strip()
    addition = (
        " Full-posterior graph transport and force-field energy scores are "
        "analysis-only and cannot rescue a failed registered exact-node gate."
    )
    output["claim_boundary"] = original_boundary + addition
    return output


def augment_contact_posterior_result_from_bundle(
    result: Mapping[str, Any],
    bundle_directory: str | Path,
    *,
    diffusion_strength: float,
) -> dict[str, Any]:
    """Load registered graph/config identities and add posterior topology scores."""

    bundle = Path(bundle_directory).resolve()
    summary_path = bundle / "summary.json"
    _require(summary_path.is_file(), "bundle summary.json is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    benchmark_config = CounterfactualBenchmarkConfig(
        **dict(summary["benchmark_config"])
    )
    contact_config = dict(summary["contact_config"])
    confidence_level = float(contact_config["confidence_level"])
    graphs = {
        protocol.graph_object.name: protocol.graph_object
        for protocol in build_protocol(benchmark_config)
    }
    return augment_contact_posterior_result(
        result,
        graphs=graphs,
        config=TopologyPosteriorScoreConfig(
            confidence_level=confidence_level,
            diffusion_strength=diffusion_strength,
        ),
    )


__all__ = [
    "TopologyPosteriorScoreConfig",
    "aggregate_posterior_topology_scores",
    "all_pairs_graph_distances",
    "assignment_graph_distance",
    "augment_contact_posterior_result",
    "augment_contact_posterior_result_from_bundle",
    "diffused_contact_field",
    "parse_contact_assignment_label",
    "parse_contact_credible_set",
    "parse_node_posterior",
    "posterior_topology_scores",
]
