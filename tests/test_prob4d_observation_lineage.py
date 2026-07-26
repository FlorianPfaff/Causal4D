from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from causal4d.observation_lineage import (
    bind_twin_belief_observation_lineage,
    compute_observation_artifact_id,
    load_observation_lineage,
)
from causal4d.prob4d_observation_lineage import (
    FIXED_EXTERNAL_CALIBRATION,
    PROB4D_JOINT_COVARIANCE_LAYOUT,
    PROB4D_JOINT_FACTOR_GROUP_SEMANTICS,
    PROB4D_LEGACY_COVARIANCE_LAYOUT,
    PROPAGATED_JOINT_GAUGE_COVARIANCE,
)


@dataclass(frozen=True)
class _Window:
    frame_stop: int
    frame_start: int = 0


@dataclass(frozen=True)
class _Context:
    case_id: str
    o_minus: _Window


@dataclass(frozen=True)
class _TwinBelief:
    context: _Context
    metadata: dict
    artifact_id: str = "f" * 64


def _artifact_parts(
    *,
    joint: bool = True,
) -> tuple[dict, dict[str, np.ndarray]]:
    factor_rank = 5 if joint else 7
    arrays = {
        "declared_frame_ids": np.asarray([1, 2, 3, 4]),
        "mean_xyz_m": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [0.0, 0.1, 1.0],
                [0.1, 0.1, 1.0],
            ]
        ),
        "frame_ids": np.asarray([1, 2, 3, 4]),
        "entity_ids": np.asarray([0, 1, 0, 1]),
        "view_indices": np.zeros(4, dtype=np.int64),
        "window_indices": np.asarray([0, 0, 1, 1]),
        "correlation_group_ids": np.asarray([0, 0, 1, 1]),
        "factor_group_ids": (
            np.zeros(4, dtype=np.int64)
            if joint
            else np.asarray([0, 0, 1, 1])
        ),
        "prior_reliability": np.asarray([0.9, 0.8, 0.7, 0.6]),
        "association_probability": np.ones(4),
        "local_covariance_m2": np.repeat(
            np.eye(3)[None],
            4,
            axis=0,
        )
        * 1e-5,
        "low_rank_factor_m": np.zeros((4, 3, factor_rank)),
        "group_ids": np.asarray([0, 1]),
        "group_prior_nominal_probability": np.asarray([0.85, 0.65]),
        "group_composite_weight": np.asarray([0.5, 0.5]),
    }
    metric_anchor: dict[str, object] = {
        "artifact_id": "a" * 64,
        "case_id": "case",
        "window_id": "window-0",
        "world_frame_id": "phystwin-world",
        "source_artifact_sha256": "1" * 64,
        "calibration_artifact_sha256": "b" * 64,
        "covariance_treatment": (
            PROPAGATED_JOINT_GAUGE_COVARIANCE
            if joint
            else FIXED_EXTERNAL_CALIBRATION
        ),
    }
    metadata: dict[str, object] = {
        "metric_coordinates": True,
        "metric_units": "m",
        "coordinate_frame": "phystwin-world",
        "metric_gauge_anchor": metric_anchor,
        "causal_source_lineage": {
            "schema_version": 1,
            "producer": "Prob4D",
            "motioncrafter_lineage_schema_version": 1,
            "motioncrafter_windowing_model": (
                "motioncrafter_sliding_window_v1"
            ),
            "source_product": "independently_decoded_overlap_windows",
            "causal_frame_stop_exclusive": 6,
            "admissibility_rule": (
                "source_frame_max < causal_frame_stop_exclusive"
            ),
            "future_prediction_payloads_opened": 0,
            "source_artifact_sha256": "c" * 64,
            "selected_windows": [
                {
                    "window_id": "window-0",
                    "source_frame_start": 0,
                    "source_frame_stop_exclusive": 3,
                    "source_frame_max": 2,
                    "frame_indices_sha256": "2" * 64,
                    "payload_sha256": "1" * 64,
                },
                {
                    "window_id": "window-1",
                    "source_frame_start": 2,
                    "source_frame_stop_exclusive": 5,
                    "source_frame_max": 4,
                    "frame_indices_sha256": "3" * 64,
                    "payload_sha256": "4" * 64,
                },
            ],
        },
    }
    if joint:
        metric_anchor.update(
            {
                "schema_name": "prob4d.metric-gauge-anchor",
                "schema_version": 2,
            }
        )
        metadata.update(
            {
                "prob4d_observation_contract_version": 2,
                "covariance_layout": PROB4D_JOINT_COVARIANCE_LAYOUT,
                "factor_group_semantics": (
                    PROB4D_JOINT_FACTOR_GROUP_SEMANTICS
                ),
                "metric_anchor_covariance_included_in_joint_factor": True,
                "joint_cross_window_gauge_covariance_represented": True,
                "gauge_posterior": {
                    "window_count": 2,
                    "exported_factor_rank": factor_rank,
                    "cross_window_covariance_preserved": True,
                },
            }
        )
    descriptor = {
        "schema_name": "phys4d.observation_belief",
        "schema_version": 1,
        "case_id": "case",
        "stream_id": "prob4d:causal-overlap-window-points",
        "causal_frame_stop": 6,
        "view_names": ["camera-0"],
        "window_names": ["window-0", "window-1"],
        "factor_names": (
            [
                f"joint_gauge_latent_{index:04d}"
                for index in range(factor_rank)
            ]
            if joint
            else [f"gauge_latent_{index}" for index in range(factor_rank)]
        ),
        "source_repository": "FlorianPfaff/Prob4D",
        "source_revision": "d" * 40,
        "source_artifact_sha256": "c" * 64,
        "metadata": metadata,
    }
    return descriptor, arrays


def _write(
    path: Path,
    *,
    joint: bool = True,
    mutate=None,
) -> None:
    descriptor, arrays = _artifact_parts(joint=joint)
    descriptor = deepcopy(descriptor)
    arrays = {name: value.copy() for name, value in arrays.items()}
    if mutate is not None:
        mutate(descriptor, arrays)
    descriptor["artifact_id"] = compute_observation_artifact_id(
        descriptor,
        arrays,
    )
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        **arrays,
    )


def test_joint_prob4d_validation_is_bound_to_twin_belief(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.npz"
    _write(path)
    lineage = load_observation_lineage(path)
    twin = _TwinBelief(_Context("case", _Window(6)), {})
    bound = bind_twin_belief_observation_lineage(twin, lineage)

    assert lineage.provider_validation["validated"] is True
    assert (
        lineage.provider_validation["covariance_layout"]
        == PROB4D_JOINT_COVARIANCE_LAYOUT
    )
    assert lineage.provider_validation["factor_rank"] == 5
    assert bound.metadata[
        "source_observation_provider_validation"
    ] == lineage.provider_validation


def test_legacy_prob4d_layout_remains_accepted(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    _write(path, joint=False)
    lineage = load_observation_lineage(path)
    assert (
        lineage.provider_validation["covariance_layout"]
        == PROB4D_LEGACY_COVARIANCE_LAYOUT
    )


def test_prob4d_lineage_rejects_changed_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del arrays
        descriptor["metadata"]["causal_source_lineage"][
            "causal_frame_stop_exclusive"
        ] = 7

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="cutoff differs"):
        load_observation_lineage(path)


def test_prob4d_lineage_rejects_future_payload_access(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del arrays
        descriptor["metadata"]["causal_source_lineage"][
            "future_prediction_payloads_opened"
        ] = 1

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="opening future payloads"):
        load_observation_lineage(path)


def test_prob4d_lineage_rejects_window_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del arrays
        descriptor["metadata"]["causal_source_lineage"][
            "selected_windows"
        ][1]["window_id"] = "another-window"

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="window order differs"):
        load_observation_lineage(path)


def test_prob4d_lineage_rejects_source_digest_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del arrays
        descriptor["metadata"]["causal_source_lineage"][
            "source_artifact_sha256"
        ] = "e" * 64

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="differs from the descriptor"):
        load_observation_lineage(path)


def test_joint_layout_rejects_factor_rank_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del arrays
        descriptor["metadata"]["gauge_posterior"][
            "exported_factor_rank"
        ] = 4

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="factor rank differs"):
        load_observation_lineage(path)


def test_joint_layout_rejects_multiple_factor_groups(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del descriptor
        arrays["factor_group_ids"] = np.asarray([0, 0, 1, 1])

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="one shared factor group"):
        load_observation_lineage(path)


def test_legacy_layout_rejects_propagated_anchor_covariance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.npz"

    def mutate(descriptor, arrays):
        del arrays
        descriptor["metadata"]["metric_gauge_anchor"][
            "covariance_treatment"
        ] = PROPAGATED_JOINT_GAUGE_COVARIANCE

    _write(path, joint=False, mutate=mutate)
    with pytest.raises(ValueError, match="requires a fixed metric anchor"):
        load_observation_lineage(path)


def test_joint_layout_rejects_missing_calibration_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del arrays
        del descriptor["metadata"]["metric_gauge_anchor"][
            "calibration_artifact_sha256"
        ]

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="calibration_artifact_sha256"):
        load_observation_lineage(path)
