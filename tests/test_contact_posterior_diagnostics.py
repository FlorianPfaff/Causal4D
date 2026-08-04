from __future__ import annotations

import numpy as np
import pytest

from causal4d.benchmark import CounterfactualBenchmarkConfig, build_protocol
from causal4d.contact_inference import ContactState
from causal4d.contact_posterior_diagnostics import (
    DiagnosticConfig,
    _aggregate_rows,
    _all_pairs_graph_distances,
    _assignment_distances,
    _diffused_force_field,
    _enrich_rows,
    _posterior_diagnostic_metrics,
    _validate_recomputed_parity,
)


def _rope():
    return next(
        protocol.graph_object
        for protocol in build_protocol(CounterfactualBenchmarkConfig())
        if protocol.graph_object.name == "rope"
    )


def _posterior_row(node_truth: str, node_map: str) -> dict[str, object]:
    return {
        "seed": 100,
        "object": "rope",
        "source_objects": "cloth;soft_block",
        "world_condition": "shifted_contact",
        "setting": "online_adaptation",
        "observation_fraction": 0.2,
        "node_truth": node_truth,
        "node_map": node_map,
        "node_correct": node_truth == node_map,
        "node_confidence": 0.9,
        "node_truth_probability": 0.7,
        "node_brier": 0.1,
        "node_credible_covered": True,
        "delay_map": 2,
        "delay_map_correct": True,
        "joint_effective_sample_size": 5.0,
        "joint_normalized_entropy": 0.4,
        "node_posterior_entropy": 0.5,
        "node_posterior_normalized_entropy": 0.45,
        "node_posterior_effective_sample_size": 1.8,
        "node_posterior_support_size": 3,
        "node_credible_set_size": 2,
        "node_posterior": '{"4":0.2,"5":0.7,"6":0.1}',
        "node_credible_set": '["5","4"]',
    }


def _interventions() -> list[dict[str, object]]:
    common = {
        "seed": 100,
        "object": "rope",
        "world_condition": "shifted_contact",
        "setting": "online_adaptation",
    }
    return [
        {**common, "method": "nominal_physics", "trajectory_rmse_m": 0.004},
        {**common, "method": "latent_contact", "trajectory_rmse_m": 0.001},
    ]


def test_graph_distance_and_force_proxy() -> None:
    graph = _rope()
    distances = _all_pairs_graph_distances(graph)
    mean_distance, maximum_distance = _assignment_distances(
        (5,), (4,), distances
    )
    truth = _diffused_force_field(graph, (5,), strength=1.0)
    exact = _diffused_force_field(graph, (5,), strength=1.0)
    neighbor = _diffused_force_field(graph, (4,), strength=1.0)

    assert mean_distance == 1.0
    assert maximum_distance == 1
    assert np.dot(truth, exact) == pytest.approx(1.0)
    assert 0.0 < float(np.dot(truth, neighbor)) < 1.0


def test_posterior_entropy_and_credible_size() -> None:
    states = (
        ContactState((5,), 1.0, 0, 0.0, 0.0),
        ContactState((4,), 1.0, 0, 0.0, 0.0),
        ContactState((6,), 1.0, 0, 0.0, 0.0),
    )
    metrics = _posterior_diagnostic_metrics(
        states,
        np.asarray([0.75, 0.20, 0.05]),
        confidence_level=0.90,
    )

    assert metrics["node_posterior_support_size"] == 3
    assert metrics["node_credible_set_size"] == 2
    assert metrics["node_posterior_effective_sample_size"] > 1.0
    assert metrics["node_posterior_entropy"] > 0.0


def test_enrichment_separates_exact_and_neighbor() -> None:
    graph = _rope()
    exact = _enrich_rows(
        [_posterior_row("5", "5")],
        _interventions(),
        {"rope": graph},
        frame_count=56,
        config=DiagnosticConfig(),
    )
    neighbor = _enrich_rows(
        [_posterior_row("5", "4")],
        _interventions(),
        {"rope": graph},
        frame_count=56,
        config=DiagnosticConfig(force_field_equivalence_threshold=0.0),
    )

    assert exact[0]["diagnostic_category"] == "exact"
    assert neighbor[0]["one_hop_patch_recovered"] is True
    assert neighbor[0]["trajectory_improved"] is True
    assert neighbor[0]["diagnostic_category"] == (
        "trajectory_equivalent_neighbor"
    )
    aggregate = _aggregate_rows([*exact, *neighbor])
    assert aggregate["exact_node_accuracy"] == 0.5
    assert aggregate["one_hop_patch_accuracy"] == 1.0


def test_recomputed_contact_identity_must_match_bundle() -> None:
    recomputed = _posterior_row("5", "5")
    retained = {
        key: str(value) if isinstance(value, bool) else value
        for key, value in recomputed.items()
    }
    report = _validate_recomputed_parity(
        [retained], [recomputed], DiagnosticConfig()
    )
    assert report["passed"] is True

    changed = dict(recomputed)
    changed["node_map"] = "4"
    changed["node_correct"] = False
    with pytest.raises(ValueError, match="node_map"):
        _validate_recomputed_parity(
            [retained], [changed], DiagnosticConfig()
        )
