"""Graph-distance primitives for unordered contact-node sets."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
import operator

import numpy as np
from scipy.optimize import linear_sum_assignment

from causal4d.immutable_array import readonly_array


ContactNodes = tuple[int, ...]
_DISTANCE_RTOL = 1e-12
_DISTANCE_ATOL = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _node_index(value: object, *, name: str) -> int:
    _require(not isinstance(value, (bool, np.bool_)), f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    return int(result)


def _canonical_contact_nodes(
    values: Iterable[object],
    *,
    node_count: int,
    name: str,
) -> ContactNodes:
    _require(
        not isinstance(values, (str, bytes)),
        f"{name} must be an iterable of node indices",
    )
    try:
        nodes = tuple(_node_index(value, name=f"{name} node") for value in values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of node indices") from error
    _require(nodes, f"{name} must contain at least one node")
    _require(len(set(nodes)) == len(nodes), f"{name} contains duplicate nodes")
    for node in nodes:
        _require(
            0 <= node < node_count,
            f"{name} node {node} is outside [0, {node_count})",
        )
    return tuple(sorted(nodes))


def _validated_distance_matrix(
    values: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    _require(
        matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.shape[0] > 0,
        "graph distances must be a nonempty square matrix",
    )
    _require(np.all(np.isfinite(matrix)), "graph distances must be finite")
    _require(np.all(matrix >= 0.0), "graph distances must be nonnegative")
    _require(
        np.allclose(
            matrix,
            matrix.T,
            rtol=_DISTANCE_RTOL,
            atol=_DISTANCE_ATOL,
        ),
        "graph distances must be symmetric",
    )
    _require(
        np.allclose(
            np.diag(matrix),
            0.0,
            rtol=0.0,
            atol=_DISTANCE_ATOL,
        ),
        "graph-distance diagonal must be zero",
    )
    canonical = np.array(matrix, dtype=float, copy=True, order="C")
    canonical[canonical == 0.0] = 0.0
    return readonly_array(canonical, dtype=float)


def all_pairs_shortest_path_distances(
    node_count: int,
    edges: Iterable[Sequence[object]],
) -> np.ndarray:
    """Return immutable unweighted distances for a connected undirected graph."""

    count = _node_index(node_count, name="node_count")
    _require(count > 0, "node_count must be positive")
    adjacency = [set() for _ in range(count)]
    for edge_index, raw_edge in enumerate(edges):
        _require(
            not isinstance(raw_edge, (str, bytes)),
            f"edge {edge_index} must contain two node indices",
        )
        try:
            edge = tuple(raw_edge)
        except TypeError as error:
            raise ValueError(
                f"edge {edge_index} must contain two node indices"
            ) from error
        _require(
            len(edge) == 2,
            f"edge {edge_index} must contain exactly two node indices",
        )
        first = _node_index(edge[0], name=f"edge {edge_index} first node")
        second = _node_index(edge[1], name=f"edge {edge_index} second node")
        _require(
            0 <= first < count and 0 <= second < count,
            f"edge {edge_index} contains an out-of-range node",
        )
        _require(first != second, f"edge {edge_index} is a self-edge")
        adjacency[first].add(second)
        adjacency[second].add(first)

    distances = np.full((count, count), np.inf, dtype=float)
    for source in range(count):
        distances[source, source] = 0.0
        frontier: deque[int] = deque([source])
        while frontier:
            node = frontier.popleft()
            candidate = distances[source, node] + 1.0
            for neighbor in sorted(adjacency[node]):
                if candidate < distances[source, neighbor]:
                    distances[source, neighbor] = candidate
                    frontier.append(neighbor)
    _require(np.all(np.isfinite(distances)), "contact graph must be connected")
    return readonly_array(distances, dtype=float)


def _minimum_bottleneck_cost(costs: np.ndarray) -> float:
    size = costs.shape[0]

    def has_complete_matching(threshold: float) -> bool:
        matched_left = [-1] * size

        def augment(right: int, visited: list[bool]) -> bool:
            for left in range(size):
                if costs[left, right] > threshold or visited[left]:
                    continue
                visited[left] = True
                if matched_left[left] < 0 or augment(
                    matched_left[left],
                    visited,
                ):
                    matched_left[left] = right
                    return True
            return False

        return all(augment(right, [False] * size) for right in range(size))

    for threshold in np.unique(costs):
        if has_complete_matching(float(threshold)):
            return float(threshold)
    raise RuntimeError("finite assignment costs did not admit a complete matching")


def _contact_costs(
    first: Iterable[object],
    second: Iterable[object],
    graph_distances: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    matrix = _validated_distance_matrix(graph_distances)
    first_nodes = _canonical_contact_nodes(
        first,
        node_count=matrix.shape[0],
        name="first contact",
    )
    second_nodes = _canonical_contact_nodes(
        second,
        node_count=matrix.shape[0],
        name="second contact",
    )
    _require(
        len(first_nodes) == len(second_nodes),
        "contact states must have equal cardinality",
    )
    return matrix[np.ix_(first_nodes, second_nodes)]


def mean_assignment_graph_distance(
    first: Iterable[object],
    second: Iterable[object],
    graph_distances: np.ndarray | Sequence[Sequence[float]],
) -> float:
    """Return minimum mean graph distance under one-to-one assignment."""

    costs = _contact_costs(first, second, graph_distances)
    rows, columns = linear_sum_assignment(costs)
    return float(np.mean(costs[rows, columns]))


def bottleneck_assignment_graph_distance(
    first: Iterable[object],
    second: Iterable[object],
    graph_distances: np.ndarray | Sequence[Sequence[float]],
) -> float:
    """Return minimum possible maximum distance under one-to-one assignment."""

    return _minimum_bottleneck_cost(_contact_costs(first, second, graph_distances))


__all__ = [
    "ContactNodes",
    "all_pairs_shortest_path_distances",
    "bottleneck_assignment_graph_distance",
    "mean_assignment_graph_distance",
]
