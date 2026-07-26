from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from causal4d.prob4d_observation_lineage import (
    validate_prob4d_causal_observation_metadata,
)


def _parts(*, explicit_version: bool) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = {
        "metric_coordinates": True,
        "metric_units": "m",
        "coordinate_frame": "phystwin-world",
        "metric_gauge_anchor": {
            "artifact_id": "a" * 64,
            "window_id": "window-0",
            "source_kind": "prefix_registration",
            "source_artifact_sha256": "1" * 64,
        },
        "gauge_mode": "sequential",
        "joint_cross_window_gauge_covariance_represented": True,
        "gauge_posterior": {
            "model": "sequential_joint_spanning_tree_v1",
            "window_count": 1,
            "full_dimension": 7,
            "exported_factor_rank": 2,
            "retained_covariance_trace_fraction": 1.0,
            "minimum_retained_gauge_trace": 0.999,
            "cross_window_covariance_preserved": True,
            "fixed_lag_boundary_covariance_is_approximate": False,
            "parent_window_ids": [None],
        },
        "causal_source_lineage": {
            "schema_version": 1,
            "producer": "Prob4D",
            "motioncrafter_lineage_schema_version": 1,
            "motioncrafter_windowing_model": "motioncrafter_sliding_window_v1",
            "source_product": "independently_decoded_overlap_windows",
            "causal_frame_stop_exclusive": 2,
            "admissibility_rule": (
                "source_frame_max < causal_frame_stop_exclusive"
            ),
            "future_prediction_payloads_opened": 0,
            "source_artifact_sha256": "c" * 64,
            "selected_windows": [
                {
                    "window_id": "window-0",
                    "source_frame_start": 0,
                    "source_frame_stop_exclusive": 2,
                    "source_frame_max": 1,
                    "frame_indices_sha256": "2" * 64,
                    "payload_sha256": "1" * 64,
                }
            ],
        },
    }
    if explicit_version:
        metadata["prob4d_causal_stream_contract_version"] = 2
        metadata["metric_gauge_anchor"].update(
            {
                "schema_name": "prob4d.metric-gauge-anchor",
                "schema_version": 1,
                "coordinate_frame": "phystwin-world",
                "metric_units": "m",
                "covariance_treatment": "fixed_external_calibration",
            }
        )
    descriptor = {
        "schema_name": "phys4d.observation_belief",
        "schema_version": 1,
        "case_id": "case",
        "stream_id": "prob4d:causal-overlap-window-points",
        "causal_frame_stop": 2,
        "view_names": ["camera-0"],
        "window_names": ["window-0"],
        "factor_names": [
            "joint_gauge_latent_0000",
            "joint_gauge_latent_0001",
        ],
        "source_repository": "FlorianPfaff/Prob4D",
        "source_revision": "d" * 40,
        "source_artifact_sha256": "c" * 64,
        "metadata": metadata,
    }
    arrays = {
        "frame_ids": np.asarray([1]),
        "window_indices": np.asarray([0]),
        "factor_group_ids": np.asarray([0]),
    }
    return descriptor, arrays


def test_explicit_prob4d_joint_stream_contract_v2_is_accepted() -> None:
    descriptor, arrays = _parts(explicit_version=True)

    validation = validate_prob4d_causal_observation_metadata(
        descriptor,
        arrays,
    )

    assert validation["stream_contract_version"] == 2
    assert validation["stream_contract_version_inferred"] is False
    assert validation["gauge_covariance_semantics"] == (
        "joint_cross_window_sim3_gauge_covariance"
    )


def test_prob4d_020_joint_stream_contract_is_inferred() -> None:
    descriptor, arrays = _parts(explicit_version=False)

    validation = validate_prob4d_causal_observation_metadata(
        descriptor,
        arrays,
    )

    assert validation["stream_contract_version"] == 2
    assert validation["stream_contract_version_inferred"] is True


def test_joint_stream_contract_rejects_approximate_fixed_lag_covariance() -> None:
    descriptor, arrays = _parts(explicit_version=True)
    descriptor = deepcopy(descriptor)
    descriptor["metadata"]["gauge_posterior"][
        "fixed_lag_boundary_covariance_is_approximate"
    ] = True

    with pytest.raises(ValueError, match="approximate fixed-lag"):
        validate_prob4d_causal_observation_metadata(descriptor, arrays)


def test_joint_stream_contract_rejects_noncanonical_factor_names() -> None:
    descriptor, arrays = _parts(explicit_version=True)
    descriptor = deepcopy(descriptor)
    descriptor["factor_names"][1] = "joint_gauge_latent_bad"

    with pytest.raises(ValueError, match="factor names are not canonical"):
        validate_prob4d_causal_observation_metadata(descriptor, arrays)
