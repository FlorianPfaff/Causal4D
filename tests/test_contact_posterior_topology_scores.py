from __future__ import annotations

import json

import numpy as np
import pytest

from causal4d.contact_posterior_topology_scores import (
    TopologyPosteriorScoreConfig,
    all_pairs_graph_distances,
    assignment_graph_distance,
    augment_contact_posterior_result,
    parse_contact_credible_set,
    parse_node_posterior,
    posterior_topology_scores,
)
from causal4d.simulator import GraphObject, PhysicalParameters


def _graph(
    *,
    name: str = "path",
    positions: np.ndarray | None = None,
    edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2)),
) -> GraphObject:
    if positions is None:
        positions = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    return GraphObject(
        name=name,
        rest_positions=positions,
        edges=edges,
        mass=1.0,
        support_stiffness=0.0,
        true_parameters=PhysicalParameters(
            stiffness=1.0,
            damping=1.0,
            contact_gain=1.0,
        ),
        sensor_nodes=(0, positions.shape[0] - 1),
    )


def _config(*, confidence_level: float = 0.8) -> TopologyPosteriorScoreConfig:
    return TopologyPosteriorScoreConfig(
        confidence_level=confidence_level,
        diffusion_strength=1.0,
    )


def test_node_posterior_and_credible_set_parsing_is_strict() -> None:
    posterior = parse_node_posterior('{"0":0.25,"1":0.75}')
    credible = parse_contact_credible_set('["1","0"]')

    assert posterior == {(0,): 0.25, (1,): 0.75}
    assert credible == ((1,), (0,))
    with pytest.raises(ValueError, match="sum to one"):
        parse_node_posterior('{"0":0.2,"1":0.2}')
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        parse_node_posterior('{"0":NaN,"1":1.0}')
    with pytest.raises(ValueError, match="duplicate assignments"):
        parse_contact_credible_set('["0","0"]')


def test_assignment_graph_distance_is_permutation_invariant() -> None:
    graph = _graph()
    distances = all_pairs_graph_distances(graph)

    assert assignment_graph_distance((0, 2), (2, 0), distances) == (0.0, 0)
    assert assignment_graph_distance((0,), (2,), distances) == (2.0, 2)


def test_full_posterior_reports_transport_mass_and_credible_radius() -> None:
    graph = _graph()
    result = posterior_topology_scores(
        graph,
        {(0,): 0.2, (1,): 0.6, (2,): 0.2},
        (0,),
        config=_config(),
        credible_set=((1,), (0,)),
    )

    assert result["posterior_expected_assignment_graph_distance_hops"] == (
        pytest.approx(1.0)
    )
    assert result["posterior_expected_max_assignment_graph_distance_hops"] == (
        pytest.approx(1.0)
    )
    assert result["posterior_exact_node_mass"] == pytest.approx(0.2)
    assert result["posterior_one_hop_patch_mass"] == pytest.approx(0.8)
    assert result["posterior_two_hop_patch_mass"] == pytest.approx(1.0)
    assert result["posterior_truth_centered_credible_radius_hops"] == 1
    assert result["node_credible_set_min_mean_graph_distance_hops"] == 0.0
    assert result["node_credible_set_one_hop_patch_covered"] is True


def test_force_field_energy_score_rewards_truth_and_calibrated_spread() -> None:
    graph = _graph()
    truth = posterior_topology_scores(
        graph,
        {(0,): 1.0},
        (0,),
        config=_config(),
    )
    mixture = posterior_topology_scores(
        graph,
        {(0,): 0.5, (1,): 0.5},
        (0,),
        config=_config(),
    )
    wrong = posterior_topology_scores(
        graph,
        {(1,): 1.0},
        (0,),
        config=_config(),
    )

    assert truth["posterior_force_field_energy_score"] == pytest.approx(0.0)
    assert 0.0 < mixture["posterior_force_field_energy_score"]
    assert (
        mixture["posterior_force_field_energy_score"]
        < (wrong["posterior_force_field_energy_score"])
    )
    assert truth["posterior_expected_force_field_cosine"] == pytest.approx(1.0)
    assert wrong["posterior_expected_force_field_cosine"] < 1.0


def test_force_field_mass_thresholds_use_the_complete_posterior() -> None:
    graph = _graph()
    result = posterior_topology_scores(
        graph,
        {(0,): 0.7, (2,): 0.3},
        (0,),
        config=_config(),
    )

    assert result["posterior_force_field_mass_at_0_8"] == pytest.approx(0.7)
    assert result["posterior_force_field_mass_at_0_9"] == pytest.approx(0.7)
    assert result["posterior_force_field_mass_at_0_95"] == pytest.approx(0.7)


def test_disconnected_graph_is_rejected() -> None:
    graph = _graph(
        name="disconnected",
        positions=np.asarray(((0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (4.0, 0.0))),
        edges=((0, 1), (2, 3)),
    )

    with pytest.raises(ValueError, match="disconnected"):
        all_pairs_graph_distances(graph)


def test_augmentation_is_additive_and_keeps_registered_gates_unchanged() -> None:
    graph = _graph()
    result = {
        "schema_version": 1,
        "claim_boundary": "Controlled diagnostic only.",
        "registered_gate_status_unchanged": True,
        "rows": [
            {
                "seed": 1,
                "object": graph.name,
                "world_condition": "shifted_contact",
                "setting": "online_adaptation",
                "node_truth": "0",
                "node_posterior": json.dumps(
                    {"0": 0.2, "1": 0.6, "2": 0.2},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "node_credible_set": '["1","0"]',
            }
        ],
    }

    augmented = augment_contact_posterior_result(
        result,
        graphs={graph.name: graph},
        config=_config(),
    )

    scores = augmented["posterior_topology_scores"]
    row = augmented["rows"][0]
    assert scores["schema_version"] == 1
    assert scores["registered_success_gates_unchanged"] is True
    assert scores["overall"]["case_count"] == 1
    assert scores["by_topology"][0]["object"] == graph.name
    assert row["posterior_one_hop_patch_mass"] == pytest.approx(0.8)
    assert augmented["registered_gate_status_unchanged"] is True
    assert "cannot rescue" in augmented["claim_boundary"]


def test_configuration_rejects_undeclared_or_unsorted_thresholds() -> None:
    with pytest.raises(ValueError, match="unique and sorted"):
        TopologyPosteriorScoreConfig(
            force_field_similarity_thresholds=(0.9, 0.8),
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        TopologyPosteriorScoreConfig(
            force_field_similarity_thresholds=(1.1,),
        )
