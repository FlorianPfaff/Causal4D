from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from causal4d.graph_contact_measure import (
    GraphContactMeasure,
    all_pairs_shortest_path_distances,
    bottleneck_assignment_graph_distance,
    mean_assignment_graph_distance,
)


def _line(node_count: int) -> np.ndarray:
    return all_pairs_shortest_path_distances(
        node_count,
        [(node, node + 1) for node in range(node_count - 1)],
    )


def test_measure_aggregates_permuted_support_deterministically() -> None:
    measure = GraphContactMeasure.from_weighted_contacts(
        [(3,), (0,), (1,), (3,)],
        np.asarray([0.1, 0.5, 0.3, 0.1]),
        _line(4),
    )

    assert measure.support == ((0,), (1,), (3,))
    assert measure.probabilities == pytest.approx((0.5, 0.3, 0.2))
    assert measure.map_state == (0,)
    assert measure.map_states == ((0,),)
    assert measure.credible_states(0.8) == ((0,), (1,))
    assert measure.node_marginal_probabilities == pytest.approx((0.5, 0.3, 0.0, 0.2))
    assert measure.entropy > 0.0
    assert 0.0 < measure.normalized_entropy < 1.0
    assert measure.effective_support_size > 1.0
    assert len(measure.graph_distances_sha256) == 64
    assert len(measure.measure_sha256) == 64
    with pytest.raises(ValueError, match="read-only"):
        measure.graph_distances[0, 1] = 100.0
    with pytest.raises(ValueError, match="WRITEABLE flag"):
        measure.graph_distances.setflags(write=True)


def test_aggregation_identity_is_independent_of_hypothesis_order() -> None:
    contacts = [
        (0,),
        (1,),
        (0,),
        (1,),
        (0,),
        (1,),
        (0,),
        (1,),
        (0,),
        (1,),
    ]
    weights = [
        0.9259554437470779,
        0.01726796332422633,
        3.0873415330523455e-09,
        1.8050451298364748e-12,
        2.0120903868730362e-07,
        1.5025737857946074e-09,
        0.056770777154068246,
        1.3944648782111336e-11,
        4.0746671855770216e-08,
        5.5692132519611315e-06,
    ]
    permutation = [5, 4, 9, 6, 0, 1, 7, 2, 8, 3]
    first = GraphContactMeasure.from_weighted_contacts(
        contacts,
        weights,
        _line(2),
    )
    second = GraphContactMeasure.from_weighted_contacts(
        [contacts[index] for index in permutation],
        [weights[index] for index in permutation],
        _line(2),
    )

    assert first.probabilities == second.probabilities
    assert first.measure_sha256 == second.measure_sha256
    assert first == second
    assert hash(first) == hash(second)


def test_equality_includes_graph_topology() -> None:
    line = _line(3)
    triangle = all_pairs_shortest_path_distances(
        3,
        [(0, 1), (1, 2), (0, 2)],
    )
    first = GraphContactMeasure.from_weighted_contacts([(0,)], [1.0], line)
    same = GraphContactMeasure.from_weighted_contacts([(0,)], [1.0], line)
    different_graph = GraphContactMeasure.from_weighted_contacts(
        [(0,)],
        [1.0],
        triangle,
    )

    assert first == same
    assert hash(first) == hash(same)
    assert first != different_graph
    assert first != object()


def test_signed_zero_has_one_canonical_identity() -> None:
    positive_graph = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    negative_graph = np.asarray([[-0.0, 1.0], [1.0, -0.0]])
    positive = GraphContactMeasure(
        ((0,), (1,)),
        (0.0, 1.0),
        positive_graph,
    )
    negative = GraphContactMeasure(
        ((0,), (1,)),
        (-0.0, 1.0),
        negative_graph,
    )

    assert positive == negative
    assert hash(positive) == hash(negative)
    assert positive.measure_sha256 == negative.measure_sha256
    assert positive.graph_distances_sha256 == negative.graph_distances_sha256


def test_record_is_strict_deterministic_json() -> None:
    measure = GraphContactMeasure.from_weighted_contacts(
        [(0,), (1,)],
        [0.75, 0.25],
        _line(2),
    )

    first = json.dumps(
        measure.as_record(0.8, truth=(1,)),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    second = json.dumps(
        measure.as_record(0.8, truth=(1,)),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert first == second


def test_direct_construction_canonicalizes_support_order_and_identity() -> None:
    distances = _line(3)
    first = GraphContactMeasure(((2,), (0,)), (0.4, 0.6), distances)
    second = GraphContactMeasure(((0,), (2,)), (0.6, 0.4), distances)

    assert first.support == ((0,), (2,))
    assert first.probabilities == pytest.approx((0.6, 0.4))
    assert first.measure_sha256 == second.measure_sha256


def test_contact_state_adapter_uses_existing_contact_nodes_contract() -> None:
    states = [
        SimpleNamespace(contact_nodes=(0,)),
        SimpleNamespace(contact_nodes=(1,)),
    ]
    measure = GraphContactMeasure.from_contact_states(
        states,
        [0.7, 0.3],
        _line(2),
    )

    assert measure.support == ((0,), (1,))
    with pytest.raises(ValueError, match="no contact_nodes"):
        GraphContactMeasure.from_contact_states(
            [object()],
            [1.0],
            _line(2),
        )


def test_graph_bayes_tie_and_expected_distances() -> None:
    measure = GraphContactMeasure.from_weighted_contacts(
        [(0,), (1,), (3,)],
        [0.5, 0.3, 0.2],
        _line(4),
    )

    assert measure.expected_mean_distance((0,)) == pytest.approx(0.9)
    assert measure.expected_pairwise_mean_distance == pytest.approx(1.14)
    assert measure.graph_bayes_risk == pytest.approx(0.9)
    assert measure.graph_bayes_states == ((0,), (1,))
    assert measure.graph_bayes_state == (0,)
    assert measure.energy_style_score((0,)) == pytest.approx(0.33)


def test_credible_region_and_radius_coverage_are_topology_aware() -> None:
    measure = GraphContactMeasure.from_weighted_contacts(
        [(0,), (1,), (3,)],
        [0.5, 0.3, 0.2],
        _line(4),
    )
    record = measure.as_record(0.8, truth=(2,))

    assert measure.credible_region_nodes(0.8) == (0, 1)
    assert measure.credible_region_nodes(0.8, radius=1.0) == (0, 1, 2)
    assert measure.credible_radius_covered((2,), 0.8, radius=0.0) is False
    assert measure.credible_radius_covered((2,), 0.8, radius=1.0) is True
    assert record["credible_exact_covered"] is False
    assert record["credible_one_hop_covered"] is True
    assert record["posterior_expected_mean_assignment_graph_distance"] == (
        pytest.approx(1.5)
    )
    assert record["truth_probability"] == 0.0
    assert record["diagnostic_only"] is True
    assert record["measure_sha256"] == measure.measure_sha256


def test_multi_contact_states_are_unordered_and_use_optimal_assignment() -> None:
    distances = _line(5)
    measure = GraphContactMeasure.from_weighted_contacts(
        [(0, 4), (4, 0), (1, 3)],
        [0.25, 0.25, 0.5],
        distances,
    )

    assert measure.support == ((0, 4), (1, 3))
    assert measure.probabilities == pytest.approx((0.5, 0.5))
    assert measure.map_states == ((0, 4), (1, 3))
    assert measure.credible_states(0.5) == ((0, 4), (1, 3))
    assert measure.credible_states(
        0.5,
        close_boundary_ties=False,
    ) == ((0, 4),)
    assert measure.mean_distance((4, 0), (1, 3)) == 1.0
    assert measure.bottleneck_distance((4, 0), (1, 3)) == 1.0
    assert mean_assignment_graph_distance((4, 0), (1, 3), distances) == 1.0
    assert (
        bottleneck_assignment_graph_distance(
            (4, 0),
            (1, 3),
            distances,
        )
        == 1.0
    )


def test_true_bottleneck_assignment_is_not_tied_to_minimum_sum_assignment() -> None:
    distances = all_pairs_shortest_path_distances(
        5,
        [(0, 3), (0, 4), (1, 0), (3, 2), (4, 3)],
    )
    measure = GraphContactMeasure.from_weighted_contacts(
        [(1, 4), (2, 4)],
        [0.5, 0.5],
        distances,
    )

    # Minimum-sum assignment uses costs (3, 0), while the bottleneck-optimal
    # assignment uses (2, 2). The two summaries therefore solve different tasks.
    assert measure.mean_distance((1, 4), (2, 4)) == 1.5
    assert measure.bottleneck_distance((1, 4), (2, 4)) == 2.0


@pytest.mark.parametrize(
    ("contacts", "weights", "message"),
    [
        ([(0,), (1, 2)], [0.5, 0.5], "equal cardinality"),
        ([(0, 0)], [1.0], "duplicate nodes"),
        ([(False,)], [1.0], "must be an integer"),
        ([(0,)], [0.9], "sum to one"),
        ([(0,)], [-1.0], "nonnegative"),
        ([(0,)], [float("nan")], "finite"),
    ],
)
def test_measure_rejects_invalid_support_or_weights(
    contacts: list[tuple[object, ...]],
    weights: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GraphContactMeasure.from_weighted_contacts(contacts, weights, _line(3))


def test_graph_validation_rejects_disconnection_and_bad_distance_matrix() -> None:
    with pytest.raises(ValueError, match="connected"):
        all_pairs_shortest_path_distances(3, [(0, 1)])
    with pytest.raises(ValueError, match="symmetric"):
        GraphContactMeasure.from_weighted_contacts(
            [(0,)],
            [1.0],
            np.asarray([[0.0, 1.0], [2.0, 0.0]]),
        )
