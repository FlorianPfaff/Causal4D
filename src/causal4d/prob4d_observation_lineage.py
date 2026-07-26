"""Independently validate strict Prob4D causal observation metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

PROB4D_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
PROB4D_CAUSAL_STREAM_ID = "prob4d:causal-overlap-window-points"
PROB4D_CAUSAL_LINEAGE_VERSION = 1
PROB4D_OBSERVATION_CONTRACT_VERSION = 2
PROB4D_JOINT_COVARIANCE_LAYOUT = "joint_sim3_tree_root_v1"
PROB4D_APPROXIMATE_FIXED_LAG_LAYOUT = (
    "approximate_fixed_lag_block_diagonal_sim3_root_v1"
)
PROB4D_JOINT_FACTOR_GROUP_SEMANTICS = (
    "single_shared_standard_normal_latent"
)
PROB4D_LEGACY_COVARIANCE_LAYOUT = "independent_window_sim3_v1"
PROB4D_GAUGE_FACTOR_NAMES = tuple(
    f"gauge_latent_{index}" for index in range(7)
)
PROB4D_METRIC_ANCHOR_SCHEMA = "prob4d.metric-gauge-anchor"
PROB4D_METRIC_ANCHOR_VERSION = 2
FIXED_EXTERNAL_CALIBRATION = "fixed_external_calibration"
PROPAGATED_JOINT_GAUGE_COVARIANCE = "propagated_joint_gauge_covariance"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, *, name: str) -> str:
    result = str(value)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return result


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _require_integer(value: Any, *, name: str) -> int:
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool),
        f"{name} must be an integer",
    )
    return int(value)


def _validate_factor_layout(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    window_names = tuple(map(str, descriptor.get("window_names", ())))
    factor_names = tuple(map(str, descriptor.get("factor_names", ())))
    factor_group_ids = np.asarray(arrays["factor_group_ids"], dtype=np.int64)
    window_indices = np.asarray(arrays["window_indices"], dtype=np.int64)
    contract_version = metadata.get("prob4d_observation_contract_version")
    covariance_layout = metadata.get("covariance_layout")
    declares_joint = contract_version is not None or covariance_layout is not None
    if not declares_joint:
        _require(
            factor_names == PROB4D_GAUGE_FACTOR_NAMES,
            "Prob4D causal artifact has changed legacy gauge factor names",
        )
        _require(
            np.array_equal(factor_group_ids, window_indices),
            "legacy Prob4D gauge factor groups must equal window indices",
        )
        return PROB4D_LEGACY_COVARIANCE_LAYOUT, window_names, factor_names

    _require(
        _require_integer(
            contract_version,
            name="Prob4D observation contract version",
        )
        == PROB4D_OBSERVATION_CONTRACT_VERSION,
        "unsupported Prob4D observation contract version",
    )
    _require(
        covariance_layout
        in {
            PROB4D_JOINT_COVARIANCE_LAYOUT,
            PROB4D_APPROXIMATE_FIXED_LAG_LAYOUT,
        },
        "unsupported Prob4D covariance layout",
    )
    _require(
        metadata.get("factor_group_semantics")
        == PROB4D_JOINT_FACTOR_GROUP_SEMANTICS,
        "Prob4D joint factor-group semantics changed",
    )
    expected_factor_names = tuple(
        f"joint_gauge_latent_{index:04d}"
        for index in range(len(factor_names))
    )
    _require(
        factor_names == expected_factor_names,
        "Prob4D joint gauge factor names changed",
    )
    _require(
        np.all(factor_group_ids == 0),
        "Prob4D joint gauge factors must use one shared factor group",
    )
    gauge_posterior = _require_mapping(
        metadata.get("gauge_posterior"),
        name="gauge_posterior",
    )
    _require(
        _require_integer(
            gauge_posterior.get("window_count"),
            name="gauge posterior window_count",
        )
        == len(window_names),
        "joint gauge posterior window count differs from the descriptor",
    )
    _require(
        _require_integer(
            gauge_posterior.get("exported_factor_rank"),
            name="gauge posterior exported_factor_rank",
        )
        == len(factor_names),
        "joint gauge posterior factor rank differs from the descriptor",
    )
    cross_window_covariance_preserved = (
        gauge_posterior.get("cross_window_covariance_preserved") is True
    )
    if covariance_layout == PROB4D_JOINT_COVARIANCE_LAYOUT:
        _require(
            cross_window_covariance_preserved,
            "joint Prob4D layout must preserve cross-window covariance",
        )
        _require(
            metadata.get("joint_cross_window_gauge_covariance_represented")
            is True,
            "Prob4D artifact does not confirm represented cross-window covariance",
        )
    else:
        _require(
            not cross_window_covariance_preserved,
            "approximate fixed-lag layout cannot claim cross-window covariance",
        )
        _require(
            gauge_posterior.get("model")
            == "fixed_lag_block_diagonal_approximation_v1",
            "approximate fixed-lag layout has changed posterior model",
        )
        _require(
            gauge_posterior.get(
                "fixed_lag_boundary_covariance_is_approximate"
            )
            is True
            or metadata.get("gauge_mode") == "fixed_lag",
            "approximate fixed-lag layout lacks its approximation declaration",
        )
        _require(
            metadata.get("joint_cross_window_gauge_covariance_represented")
            is False,
            "approximate fixed-lag layout cannot represent joint covariance",
        )
    _require(
        metadata.get("metric_anchor_covariance_included_in_joint_factor")
        is True,
        "Prob4D joint factor does not include metric-anchor covariance",
    )
    return str(covariance_layout), window_names, factor_names


def is_prob4d_causal_observation_descriptor(
    descriptor: Mapping[str, Any],
) -> bool:
    """Return whether the descriptor declares the strict Prob4D stream."""

    return (
        descriptor.get("source_repository") == PROB4D_SOURCE_REPOSITORY
        and descriptor.get("stream_id") == PROB4D_CAUSAL_STREAM_ID
    )


def validate_prob4d_causal_observation_metadata(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Fail closed on inconsistent metric, gauge, or temporal lineage."""

    _require(
        is_prob4d_causal_observation_descriptor(descriptor),
        "observation descriptor is not the strict Prob4D causal stream",
    )
    source_revision = str(descriptor.get("source_revision", ""))
    _require(
        bool(source_revision) and source_revision.lower() != "unknown",
        "Prob4D causal artifact has no exact source revision",
    )
    source_artifact_sha256 = _require_sha256(
        descriptor.get("source_artifact_sha256", ""),
        name="source_artifact_sha256",
    )
    metadata = _require_mapping(
        descriptor.get("metadata"),
        name="observation metadata",
    )
    covariance_layout, window_names, factor_names = _validate_factor_layout(
        descriptor,
        arrays,
        metadata,
    )
    _require(bool(window_names), "Prob4D causal artifact has no source windows")

    _require(
        metadata.get("metric_coordinates") is True,
        "Prob4D causal artifact must declare metric coordinates",
    )
    _require(
        metadata.get("metric_units") == "m",
        "Prob4D causal artifact must declare metric units",
    )
    coordinate_frame = str(metadata.get("coordinate_frame", ""))
    _require(
        bool(coordinate_frame),
        "Prob4D causal artifact has no coordinate frame",
    )

    anchor = _require_mapping(
        metadata.get("metric_gauge_anchor"),
        name="metric_gauge_anchor",
    )
    anchor_id = _require_sha256(
        anchor.get("artifact_id", ""),
        name="metric gauge-anchor artifact_id",
    )
    anchor_source_sha256 = _require_sha256(
        anchor.get("source_artifact_sha256", ""),
        name="metric gauge-anchor source_artifact_sha256",
    )
    _require_sha256(
        anchor.get("calibration_artifact_sha256", ""),
        name="metric gauge-anchor calibration_artifact_sha256",
    )
    _require(
        anchor.get("window_id") == window_names[0],
        "metric gauge anchor does not identify the first window",
    )
    _require(
        anchor.get("world_frame_id") == coordinate_frame,
        "metric gauge-anchor frame differs from observation frame",
    )
    if anchor.get("case_id") is not None:
        _require(
            anchor.get("case_id") == descriptor.get("case_id"),
            "metric gauge-anchor case differs from observation case",
        )
    covariance_treatment = str(anchor.get("covariance_treatment", ""))
    if covariance_layout in {
        PROB4D_JOINT_COVARIANCE_LAYOUT,
        PROB4D_APPROXIMATE_FIXED_LAG_LAYOUT,
    }:
        _require(
            anchor.get("schema_name") == PROB4D_METRIC_ANCHOR_SCHEMA,
            "Prob4D joint artifact has an unsupported metric-anchor schema",
        )
        _require(
            _require_integer(
                anchor.get("schema_version"),
                name="metric gauge-anchor schema_version",
            )
            == PROB4D_METRIC_ANCHOR_VERSION,
            "Prob4D joint artifact has an unsupported metric-anchor version",
        )
        if covariance_layout == PROB4D_JOINT_COVARIANCE_LAYOUT:
            _require(
                covariance_treatment
                in {
                    FIXED_EXTERNAL_CALIBRATION,
                    PROPAGATED_JOINT_GAUGE_COVARIANCE,
                },
                "Prob4D joint artifact has an unsupported anchor covariance treatment",
            )
        else:
            _require(
                covariance_treatment == FIXED_EXTERNAL_CALIBRATION,
                "approximate fixed-lag artifact requires a fixed metric anchor",
            )
    else:
        _require(
            covariance_treatment == FIXED_EXTERNAL_CALIBRATION,
            "legacy Prob4D artifact requires a fixed metric anchor",
        )

    causal_stop = _require_integer(
        descriptor.get("causal_frame_stop"),
        name="observation causal_frame_stop",
    )
    lineage = _require_mapping(
        metadata.get("causal_source_lineage"),
        name="causal_source_lineage",
    )
    _require(
        _require_integer(
            lineage.get("schema_version"),
            name="causal lineage schema_version",
        )
        == PROB4D_CAUSAL_LINEAGE_VERSION,
        "unsupported Prob4D causal-lineage version",
    )
    _require(
        lineage.get("producer") == "Prob4D",
        "causal lineage has changed producer",
    )
    _require(
        lineage.get("motioncrafter_lineage_schema_version") == 1,
        "unsupported MotionCrafter temporal-lineage version",
    )
    _require(
        lineage.get("motioncrafter_windowing_model")
        == "motioncrafter_sliding_window_v1",
        "unsupported MotionCrafter windowing model",
    )
    _require(
        lineage.get("source_product")
        == "independently_decoded_overlap_windows",
        "Prob4D causal artifact uses an inadmissible source product",
    )
    _require(
        _require_integer(
            lineage.get("causal_frame_stop_exclusive"),
            name="causal lineage frame stop",
        )
        == causal_stop,
        "causal lineage cutoff differs from the artifact cutoff",
    )
    _require(
        _require_integer(
            lineage.get("future_prediction_payloads_opened"),
            name="future_prediction_payloads_opened",
        )
        == 0,
        "Prob4D causal artifact reports opening future payloads",
    )
    _require(
        lineage.get("admissibility_rule")
        == "source_frame_max < causal_frame_stop_exclusive",
        "Prob4D causal artifact has changed its admission rule",
    )
    lineage_source_sha256 = _require_sha256(
        lineage.get("source_artifact_sha256", ""),
        name="causal lineage source_artifact_sha256",
    )
    _require(
        lineage_source_sha256 == source_artifact_sha256,
        "causal lineage source digest differs from the descriptor",
    )

    selected = lineage.get("selected_windows")
    _require(
        isinstance(selected, list) and len(selected) == len(window_names),
        "causal lineage must identify every observation window",
    )
    window_indices = np.asarray(arrays["window_indices"], dtype=np.int64)
    frame_ids = np.asarray(arrays["frame_ids"], dtype=np.int64)
    for window_index, expected_window_id in enumerate(window_names):
        record = _require_mapping(
            selected[window_index],
            name=f"selected_windows[{window_index}]",
        )
        _require(
            record.get("window_id") == expected_window_id,
            "causal lineage window order differs from the descriptor",
        )
        start = _require_integer(
            record.get("source_frame_start"),
            name=f"selected window {expected_window_id} start",
        )
        stop = _require_integer(
            record.get("source_frame_stop_exclusive"),
            name=f"selected window {expected_window_id} stop",
        )
        maximum = _require_integer(
            record.get("source_frame_max"),
            name=f"selected window {expected_window_id} maximum",
        )
        _require(
            0 <= start <= maximum < stop <= causal_stop,
            "selected Prob4D window crosses its causal boundary",
        )
        _require_sha256(
            record.get("frame_indices_sha256", ""),
            name=f"selected window {expected_window_id} frame digest",
        )
        payload_digest = _require_sha256(
            record.get("payload_sha256", ""),
            name=f"selected window {expected_window_id} payload digest",
        )
        if window_index == 0:
            _require(
                payload_digest == anchor_source_sha256,
                "metric anchor is not bound to the first selected payload",
            )
        rows = window_indices == window_index
        if np.any(rows):
            row_frames = frame_ids[rows]
            _require(
                np.all((row_frames >= start) & (row_frames <= maximum)),
                "observation rows exceed their declared source window",
            )

    return {
        "validated": True,
        "schema_version": PROB4D_CAUSAL_LINEAGE_VERSION,
        "prob4d_observation_contract_version": (
            PROB4D_OBSERVATION_CONTRACT_VERSION
            if covariance_layout
            in {
                PROB4D_JOINT_COVARIANCE_LAYOUT,
                PROB4D_APPROXIMATE_FIXED_LAG_LAYOUT,
            }
            else 1
        ),
        "covariance_layout": covariance_layout,
        "factor_rank": len(factor_names),
        "causal_frame_stop": causal_stop,
        "window_count": len(window_names),
        "metric_anchor_id": anchor_id,
        "metric_anchor_covariance_treatment": covariance_treatment,
        "source_artifact_sha256": source_artifact_sha256,
    }


__all__ = [
    "FIXED_EXTERNAL_CALIBRATION",
    "PROB4D_APPROXIMATE_FIXED_LAG_LAYOUT",
    "PROB4D_CAUSAL_LINEAGE_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_GAUGE_FACTOR_NAMES",
    "PROB4D_JOINT_COVARIANCE_LAYOUT",
    "PROB4D_JOINT_FACTOR_GROUP_SEMANTICS",
    "PROB4D_LEGACY_COVARIANCE_LAYOUT",
    "PROB4D_OBSERVATION_CONTRACT_VERSION",
    "PROB4D_SOURCE_REPOSITORY",
    "PROPAGATED_JOINT_GAUGE_COVARIANCE",
    "is_prob4d_causal_observation_descriptor",
    "validate_prob4d_causal_observation_metadata",
]
