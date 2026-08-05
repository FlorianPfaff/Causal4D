"""Explicitly trusted migration from legacy PhysTwin target pickles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from causal4d._held_out_target_contract import (
    require_mapping,
    target_validity,
    validate_sha256,
)
from causal4d.contracts import PhysicalPosterior
from causal4d.held_out_target import HeldOutPhysicalTarget
from causal4d.trusted_pickle import load_trusted_pickle


def import_legacy_physical_target(
    posterior: PhysicalPosterior,
    final_data_pickle: str | Path,
    *,
    allow_unsafe_pickle: bool = False,
    expected_sha256: str | None = None,
    source_revision: str = "legacy-final-data-v1",
    source_artifact_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HeldOutPhysicalTarget:
    """Convert one trusted legacy ``final_data.pkl`` into a safe target artifact."""

    if not isinstance(posterior, PhysicalPosterior):
        raise TypeError("posterior must be a PhysicalPosterior")
    if expected_sha256 is None:
        raise ValueError("expected_sha256 is required for legacy target import")
    expected = validate_sha256(expected_sha256, name="expected_sha256")
    values = load_trusted_pickle(
        final_data_pickle,
        allow_unsafe_pickle=allow_unsafe_pickle,
        expected_sha256=expected,
    )
    mapping = require_mapping(values, name="legacy final_data pickle")
    required = {
        "object_points",
        "object_visibilities",
        "object_motions_valid",
    }
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError("legacy final_data pickle is missing: " + ", ".join(missing))

    raw_positions = np.asarray(mapping["object_points"])
    if raw_positions.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("legacy object_points must contain real numeric values")
    observed = raw_positions.astype(np.float64, copy=False)
    if observed.ndim != 3 or observed.shape[2] != 3:
        raise ValueError("legacy object_points must have shape (T, N, 3)")
    valid = target_validity(
        np.asarray(mapping["object_visibilities"]),
        np.asarray(mapping["object_motions_valid"]),
    )
    if valid.shape != observed.shape[:2]:
        raise ValueError("legacy target validity does not match object_points")

    frame_start = posterior.context.o_minus.frame_stop - 1
    trajectory_length = posterior.readout_trajectories_m.shape[1]
    point_count = posterior.readout_trajectories_m.shape[2]
    frame_stop = frame_start + trajectory_length
    if frame_stop != posterior.context.u_cf.frame_stop:
        raise ValueError(
            "posterior readout length does not match its counterfactual context"
        )
    if frame_stop > len(observed) or point_count > observed.shape[1]:
        raise ValueError("legacy final_data does not cover the posterior target")

    supplied_metadata = {} if metadata is None else dict(metadata)
    if "legacy_import" in supplied_metadata:
        raise ValueError("metadata key 'legacy_import' is reserved")
    supplied_metadata["legacy_import"] = {
        "format": "python-pickle",
        "object_points_key": "object_points",
        "visibility_key": "object_visibilities",
        "motion_valid_key": "object_motions_valid",
        "target_validity_semantics": "bayesian_phystwin.target_validity_v1",
    }
    target = HeldOutPhysicalTarget(
        context=posterior.context,
        source_query_id=posterior.source_query_id,
        trajectory_frame_start=frame_start,
        node_indices=np.arange(point_count, dtype=np.int64),
        positions_m=observed[frame_start:frame_stop, :point_count],
        validity_mask=valid[frame_start:frame_stop, :point_count],
        source_kind="trusted_legacy_final_data_pickle",
        source_revision=source_revision,
        source_content_sha256=expected,
        source_artifact_id=source_artifact_id,
        metadata=supplied_metadata,
    )
    target.require_compatible_physical_posterior(posterior)
    return target


__all__ = ["import_legacy_physical_target"]
