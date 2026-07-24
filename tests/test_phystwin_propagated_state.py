from __future__ import annotations

import json

import numpy as np

from causal4d.phystwin_propagated_state import (
    GUARDED_PROPAGATED,
    PROPAGATED_METHODS,
    RAW_PROPAGATED,
    _global_trajectory,
    _prediction_variance_m2,
    aggregate_guarded_propagated_state_cases,
)


def test_prediction_variance_includes_state_bias_cross_covariance() -> None:
    response = np.zeros((2, 3, 3, 1))
    response[:, :, 0, 0] = 2.0
    basis = np.ones((3, 1))
    covariance = np.zeros((4, 4))
    covariance[0, 0] = 4.0
    covariance[1, 1] = 9.0
    covariance[0, 1] = covariance[1, 0] = 1.5

    variance = _prediction_variance_m2(response, basis, covariance)

    # Var(2*s + b) = 4*4 + 9 + 4*1.5.
    np.testing.assert_allclose(variance[:, :, 0], 31.0)
    np.testing.assert_array_equal(variance[:, :, 1:], 0.0)


def test_global_trajectory_preserves_prefix_and_uses_particle_mixture() -> None:
    baseline = np.arange(30, dtype=np.float32).reshape(5, 2, 3)
    particles = np.stack((np.ones((3, 2, 3)), 3.0 * np.ones((3, 2, 3))))

    result = _global_trajectory(
        baseline,
        particles,
        np.asarray((0.25, 0.75)),
        endpoint_frame=2,
    )

    np.testing.assert_array_equal(result[:2], baseline[:2])
    np.testing.assert_allclose(result[2:], 2.5)


def test_aggregate_stops_when_every_prefix_guard_rejects(tmp_path) -> None:
    paths = []
    for index in range(2):
        methods = {}
        for method in PROPAGATED_METHODS:
            value = 0.012 if method == RAW_PROPAGATED else 0.01
            methods[method] = {
                "future": {
                    "chamfer_distance_m": {"candidate_mean_m": value},
                    "track_error_m": {"candidate_mean_m": value + 0.005},
                },
                "coverage": {"coordinate_coverage_90": 0.8},
            }
        summary = {
            "experiment": "phystwin_guarded_action_propagated_state_v1",
            "case": f"case_{index}",
            "status": "released_case_implementation_diagnostic_not_model_selection",
            "selection": {
                "accepted_state_update": False,
                "reason": "prefix-validation-regret-guard",
                "validation_improvement_fraction": -0.1,
            },
            "readout_correction_shrinkage_fraction": 0.9,
            "exact_fallback": {
                "particle_bytes_identical": True,
                "global_bytes_identical": True,
            },
            "methods": methods,
        }
        path = tmp_path / f"case_{index}.json"
        path.write_text(json.dumps(summary))
        paths.append(path)

    result = aggregate_guarded_propagated_state_cases(
        paths,
        tmp_path / "aggregate.json",
    )

    assert not result["decision"]["run_exploratory_19_case_cohort"]
    assert (
        result["comparison_vs_frozen_graph_persistence"][
            "guarded_state_acceptance_count"
        ]
        == 0
    )
    assert (
        result["methods"][GUARDED_PROPAGATED]["chamfer_distance_m"][
            "case_balanced_mean"
        ]
        == 0.01
    )

