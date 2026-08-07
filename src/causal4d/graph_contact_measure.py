"""Immutable, topology-aware summaries of graph-valued contact posteriors.

This module summarizes an already computed posterior. It does not change the frozen
Causal4D estimator, likelihood, intervention bank, or registered thresholds.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from causal4d.graph_contact_distance import (
    ContactNodes,
    _canonical_contact_nodes,
    _minimum_bottleneck_cost,
    _require,
    _validated_distance_matrix,
    all_pairs_shortest_path_distances,
    bottleneck_assignment_graph_distance,
    mean_assignment_graph_distance,
)


GRAPH_CONTACT_MEASURE_SCHEMA_VERSION = 1
GRAPH_CONTACT_MEASURE_ARTIFACT_KIND = "Causal4DGraphContactMeasure"
_PROBABILITY_RTOL = 1e-10
_PROBABILITY_ATOL = 1e-12
_TIE_RTOL = 1e-12
_TIE_ATOL = 1e-15


def _stable_sum(values: Iterable[float]) -> float:
    """Return a permutation-invariant finite sum for nonnegative probabilities."""

    return math.fsum(sorted(float(value) for value in values))


@dataclass(frozen=True, eq=False)
class GraphContactMeasure:
    """Posterior measure over equal-cardinality, unordered graph contacts."""

    support: tuple[ContactNodes, ...]
    probabilities: tuple[float, ...]
    graph_distances: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        matrix = _validated_distance_matrix(self.graph_distances)
        _require(self.support, "contact posterior support must be nonempty")
        _require(
            len(self.support) == len(self.probabilities),
            "contact probabilities must align with support",
        )
        canonical = tuple(
            _canonical_contact_nodes(
                state,
                node_count=matrix.shape[0],
                name=f"support state {index}",
            )
            for index, state in enumerate(self.support)
        )
        _require(
            len(set(canonical)) == len(canonical),
            "contact posterior support contains duplicate states",
        )
        cardinality = len(canonical[0])
        _require(
            all(len(state) == cardinality for state in canonical),
            "contact posterior states must have equal cardinality",
        )
        values = np.asarray(self.probabilities, dtype=float)
        _require(
            values.shape == (len(canonical),),
            "contact probabilities must be one-dimensional",
        )
        _require(np.all(np.isfinite(values)), "contact probabilities must be finite")
        _require(
            np.all(values >= 0.0),
            "contact probabilities must be nonnegative",
        )
        ordered = sorted(zip(canonical, values, strict=True))
        canonical = tuple(state for state, _ in ordered)
        values = tuple(float(probability) for _, probability in ordered)
        total = _stable_sum(values)
        _require(
            math.isclose(
                total,
                1.0,
                rel_tol=_PROBABILITY_RTOL,
                abs_tol=_PROBABILITY_ATOL,
            ),
            "contact probabilities must sum to one",
        )
        object.__setattr__(self, "support", canonical)
        normalized = tuple(value / total for value in values)
        object.__setattr__(
            self,
            "probabilities",
            tuple(0.0 if value == 0.0 else value for value in normalized),
        )
        object.__setattr__(self, "graph_distances", matrix)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphContactMeasure):
            return False
        return bool(
            self.support == other.support
            and self.probabilities == other.probabilities
            and np.array_equal(self.graph_distances, other.graph_distances)
        )

    def __hash__(self) -> int:
        return int(self.measure_sha256[:16], 16)

    @classmethod
    def from_weighted_contacts(
        cls,
        contacts: Sequence[Iterable[object]],
        weights: Sequence[float] | np.ndarray,
        graph_distances: np.ndarray | Sequence[Sequence[float]],
    ) -> GraphContactMeasure:
        """Canonicalize and aggregate weighted contact hypotheses."""

        matrix = _validated_distance_matrix(graph_distances)
        raw_contacts = tuple(contacts)
        values = np.asarray(weights, dtype=float)
        _require(raw_contacts, "contact posterior support must be nonempty")
        _require(
            values.shape == (len(raw_contacts),),
            "contact weights must align with support",
        )
        _require(np.all(np.isfinite(values)), "contact weights must be finite")
        _require(np.all(values >= 0.0), "contact weights must be nonnegative")
        total = _stable_sum(values)
        _require(
            math.isclose(
                total,
                1.0,
                rel_tol=_PROBABILITY_RTOL,
                abs_tol=_PROBABILITY_ATOL,
            ),
            "contact weights must sum to one",
        )
        grouped: defaultdict[ContactNodes, list[float]] = defaultdict(list)
        cardinality: int | None = None
        for index, (contact, weight) in enumerate(
            zip(raw_contacts, values, strict=True)
        ):
            state = _canonical_contact_nodes(
                contact,
                node_count=matrix.shape[0],
                name=f"contact state {index}",
            )
            cardinality = len(state) if cardinality is None else cardinality
            _require(
                len(state) == cardinality,
                "contact posterior states must have equal cardinality",
            )
            grouped[state].append(float(weight))
        support = tuple(sorted(grouped))
        return cls(
            support,
            tuple(_stable_sum(grouped[state]) / total for state in support),
            matrix,
        )

    @classmethod
    def from_contact_states(
        cls,
        states: Sequence[object],
        weights: Sequence[float] | np.ndarray,
        graph_distances: np.ndarray | Sequence[Sequence[float]],
    ) -> GraphContactMeasure:
        """Build from Causal4D states exposing a ``contact_nodes`` attribute."""

        contacts: list[Iterable[object]] = []
        for index, state in enumerate(states):
            try:
                contacts.append(getattr(state, "contact_nodes"))
            except AttributeError as error:
                raise ValueError(
                    f"contact state {index} has no contact_nodes attribute"
                ) from error
        return cls.from_weighted_contacts(contacts, weights, graph_distances)

    @property
    def node_count(self) -> int:
        return int(self.graph_distances.shape[0])

    @property
    def contact_cardinality(self) -> int:
        return len(self.support[0])

    @property
    def probability_map(self) -> Mapping[ContactNodes, float]:
        return MappingProxyType(
            dict(zip(self.support, self.probabilities, strict=True))
        )

    def _state(self, values: Iterable[object], *, name: str) -> ContactNodes:
        state = _canonical_contact_nodes(
            values,
            node_count=self.node_count,
            name=name,
        )
        _require(
            len(state) == self.contact_cardinality,
            f"{name} must have cardinality {self.contact_cardinality}",
        )
        return state

    def mean_distance(
        self,
        first: Iterable[object],
        second: Iterable[object],
    ) -> float:
        return mean_assignment_graph_distance(
            self._state(first, name="first contact"),
            self._state(second, name="second contact"),
            self.graph_distances,
        )

    def bottleneck_distance(
        self,
        first: Iterable[object],
        second: Iterable[object],
    ) -> float:
        first_state = self._state(first, name="first contact")
        second_state = self._state(second, name="second contact")
        costs = self.graph_distances[np.ix_(first_state, second_state)]
        return _minimum_bottleneck_cost(costs)

    @property
    def entropy(self) -> float:
        positive = np.asarray(
            [value for value in self.probabilities if value > 0.0],
            dtype=float,
        )
        return -float(np.sum(positive * np.log(positive)))

    @property
    def normalized_entropy(self) -> float:
        return (
            float(self.entropy / math.log(len(self.support)))
            if len(self.support) > 1
            else 0.0
        )

    @property
    def effective_support_size(self) -> float:
        values = np.asarray(self.probabilities)
        return float(1.0 / np.sum(np.square(values)))

    @property
    def graph_distances_sha256(self) -> str:
        canonical = np.asarray(self.graph_distances, dtype="<f8", order="C")
        digest = hashlib.sha256()
        digest.update(json.dumps(list(canonical.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()

    @property
    def measure_sha256(self) -> str:
        payload = {
            "schema_version": GRAPH_CONTACT_MEASURE_SCHEMA_VERSION,
            "artifact_kind": GRAPH_CONTACT_MEASURE_ARTIFACT_KIND,
            "support": [list(state) for state in self.support],
            "probabilities": list(self.probabilities),
            "graph_distances_sha256": self.graph_distances_sha256,
        }
        data = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    @property
    def map_probability(self) -> float:
        return max(self.probabilities)

    @property
    def map_states(self) -> tuple[ContactNodes, ...]:
        return tuple(
            state
            for state, probability in zip(
                self.support,
                self.probabilities,
                strict=True,
            )
            if math.isclose(
                probability,
                self.map_probability,
                rel_tol=_TIE_RTOL,
                abs_tol=_TIE_ATOL,
            )
        )

    @property
    def map_state(self) -> ContactNodes:
        return self.map_states[0]

    def credible_states(
        self,
        confidence_level: float,
        *,
        close_boundary_ties: bool = True,
    ) -> tuple[ContactNodes, ...]:
        confidence = float(confidence_level)
        _require(
            math.isfinite(confidence) and 0.0 < confidence < 1.0,
            "confidence_level must be finite and in (0, 1)",
        )
        ordered = sorted(
            zip(self.support, self.probabilities, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        selected: list[ContactNodes] = []
        cumulative = 0.0
        boundary: float | None = None
        for state, probability in ordered:
            selected.append(state)
            cumulative += probability
            if cumulative >= confidence:
                boundary = probability
                break
        if close_boundary_ties and boundary is not None:
            for state, probability in ordered[len(selected) :]:
                if not math.isclose(
                    probability,
                    boundary,
                    rel_tol=_TIE_RTOL,
                    abs_tol=_TIE_ATOL,
                ):
                    break
                selected.append(state)
        return tuple(selected)

    def credible_region_nodes(
        self,
        confidence_level: float,
        *,
        radius: float = 0.0,
        close_boundary_ties: bool = True,
    ) -> tuple[int, ...]:
        radius_value = float(radius)
        _require(
            math.isfinite(radius_value) and radius_value >= 0.0,
            "credible-region radius must be finite and nonnegative",
        )
        states = self.credible_states(
            confidence_level,
            close_boundary_ties=close_boundary_ties,
        )
        seeds = sorted({node for state in states for node in state})
        nearest = np.min(self.graph_distances[:, seeds], axis=1)
        return tuple(int(node) for node in np.flatnonzero(nearest <= radius_value))

    def credible_radius_covered(
        self,
        truth: Iterable[object],
        confidence_level: float,
        *,
        radius: float,
        close_boundary_ties: bool = True,
    ) -> bool:
        truth_state = self._state(truth, name="truth contact")
        radius_value = float(radius)
        _require(
            math.isfinite(radius_value) and radius_value >= 0.0,
            "credible-coverage radius must be finite and nonnegative",
        )
        return any(
            self.bottleneck_distance(state, truth_state) <= radius_value
            for state in self.credible_states(
                confidence_level,
                close_boundary_ties=close_boundary_ties,
            )
        )

    @property
    def node_marginal_probabilities(self) -> tuple[float, ...]:
        values = np.zeros(self.node_count)
        for state, probability in zip(
            self.support,
            self.probabilities,
            strict=True,
        ):
            values[list(state)] += probability
        return tuple(float(value) for value in values)

    def expected_mean_distance(self, reference: Iterable[object]) -> float:
        reference_state = self._state(reference, name="reference contact")
        return float(
            sum(
                probability * self.mean_distance(state, reference_state)
                for state, probability in zip(
                    self.support,
                    self.probabilities,
                    strict=True,
                )
            )
        )

    def expected_bottleneck_distance(self, reference: Iterable[object]) -> float:
        reference_state = self._state(reference, name="reference contact")
        return float(
            sum(
                probability * self.bottleneck_distance(state, reference_state)
                for state, probability in zip(
                    self.support,
                    self.probabilities,
                    strict=True,
                )
            )
        )

    @property
    def expected_pairwise_mean_distance(self) -> float:
        total = 0.0
        for first_index, first in enumerate(self.support):
            for second_index in range(first_index + 1, len(self.support)):
                total += (
                    2.0
                    * self.probabilities[first_index]
                    * self.probabilities[second_index]
                    * self.mean_distance(first, self.support[second_index])
                )
        return float(total)

    def _candidate_risks(self) -> tuple[float, ...]:
        return tuple(self.expected_mean_distance(state) for state in self.support)

    @property
    def graph_bayes_risk(self) -> float:
        return min(self._candidate_risks())

    @property
    def graph_bayes_states(self) -> tuple[ContactNodes, ...]:
        risks = self._candidate_risks()
        minimum = min(risks)
        return tuple(
            state
            for state, risk in zip(self.support, risks, strict=True)
            if math.isclose(
                risk,
                minimum,
                rel_tol=_TIE_RTOL,
                abs_tol=_TIE_ATOL,
            )
        )

    @property
    def graph_bayes_state(self) -> ContactNodes:
        return self.graph_bayes_states[0]

    def energy_style_score(self, truth: Iterable[object]) -> float:
        """Return E[d(Z, truth)] - 0.5 E[d(Z, Z')]."""

        return float(
            self.expected_mean_distance(truth)
            - 0.5 * self.expected_pairwise_mean_distance
        )

    def as_record(
        self,
        confidence_level: float,
        *,
        truth: Iterable[object] | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic JSON-compatible diagnostic record."""

        credible = self.credible_states(confidence_level)
        region = self.credible_region_nodes(confidence_level)
        one_hop = self.credible_region_nodes(confidence_level, radius=1.0)
        record: dict[str, Any] = {
            "schema_version": GRAPH_CONTACT_MEASURE_SCHEMA_VERSION,
            "artifact_kind": GRAPH_CONTACT_MEASURE_ARTIFACT_KIND,
            "measure_sha256": self.measure_sha256,
            "graph_distances_sha256": self.graph_distances_sha256,
            "diagnostic_only": True,
            "contact_cardinality": self.contact_cardinality,
            "support_size": len(self.support),
            "entropy": self.entropy,
            "normalized_entropy": self.normalized_entropy,
            "effective_support_size": self.effective_support_size,
            "support": [
                {"contact_nodes": list(state), "probability": probability}
                for state, probability in zip(
                    self.support,
                    self.probabilities,
                    strict=True,
                )
            ],
            "map_contact_nodes": list(self.map_state),
            "map_contact_states": [list(state) for state in self.map_states],
            "map_probability": self.map_probability,
            "credible_contact_states": [list(state) for state in credible],
            "credible_contact_state_count": len(credible),
            "credible_region_nodes": list(region),
            "credible_region_node_count": len(region),
            "one_hop_credible_region_nodes": list(one_hop),
            "one_hop_credible_region_node_count": len(one_hop),
            "graph_bayes_contact_nodes": list(self.graph_bayes_state),
            "graph_bayes_contact_states": [
                list(state) for state in self.graph_bayes_states
            ],
            "graph_bayes_risk": self.graph_bayes_risk,
            "expected_pairwise_mean_graph_distance": (
                self.expected_pairwise_mean_distance
            ),
            "node_marginal_probabilities": list(self.node_marginal_probabilities),
        }
        if truth is None:
            return record
        truth_state = self._state(truth, name="truth contact")
        record.update(
            {
                "truth_contact_nodes": list(truth_state),
                "truth_probability": self.probability_map.get(truth_state, 0.0),
                "map_mean_assignment_graph_distance": self.mean_distance(
                    self.map_state,
                    truth_state,
                ),
                "map_bottleneck_assignment_graph_distance": (
                    self.bottleneck_distance(self.map_state, truth_state)
                ),
                "graph_bayes_mean_assignment_graph_distance": (
                    self.mean_distance(self.graph_bayes_state, truth_state)
                ),
                "posterior_expected_mean_assignment_graph_distance": (
                    self.expected_mean_distance(truth_state)
                ),
                "posterior_expected_bottleneck_assignment_graph_distance": (
                    self.expected_bottleneck_distance(truth_state)
                ),
                "credible_exact_covered": self.credible_radius_covered(
                    truth_state,
                    confidence_level,
                    radius=0.0,
                ),
                "credible_one_hop_covered": self.credible_radius_covered(
                    truth_state,
                    confidence_level,
                    radius=1.0,
                ),
                "graph_energy_style_score": self.energy_style_score(truth_state),
            }
        )
        return record


__all__ = [
    "ContactNodes",
    "GRAPH_CONTACT_MEASURE_ARTIFACT_KIND",
    "GRAPH_CONTACT_MEASURE_SCHEMA_VERSION",
    "GraphContactMeasure",
    "all_pairs_shortest_path_distances",
    "bottleneck_assignment_graph_distance",
    "mean_assignment_graph_distance",
]
