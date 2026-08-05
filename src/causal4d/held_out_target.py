"""Portable, content-addressed held-out targets for physical evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from causal4d._held_out_target_contract import (
    HELD_OUT_PHYSICAL_TARGET_SCHEMA,
    HELD_OUT_PHYSICAL_TARGET_SCHEMA_VERSION,
    NODE_ORDER_DENSE_PREFIX_V1,
    canonical_json,
    reject_duplicate_json_keys,
    reject_nonfinite_json_constant,
    require_mapping,
    require_nonempty_string,
    require_optional_string,
    require_integer,
    validate_sha256,
    validate_target_descriptor,
)
from causal4d.atomic_io import atomic_write_binary
from causal4d.contracts import CausalContext, PhysicalPosterior, array_sha256
from causal4d.immutable_array import readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping

_ARCHIVE_FIELDS = frozenset(
    {"descriptor_json", "node_indices", "positions_m", "validity_mask"}
)


@dataclass(frozen=True)
class HeldOutPhysicalTarget:
    """Exact held-out trajectory and validity bound to one causal query."""

    context: CausalContext
    source_query_id: str
    trajectory_frame_start: int
    node_indices: np.ndarray
    positions_m: np.ndarray
    validity_mask: np.ndarray
    source_kind: str
    source_revision: str
    source_content_sha256: str
    source_artifact_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.context) is not CausalContext:
            raise ValueError("context must be a CausalContext")
        source_query_id = validate_sha256(
            self.source_query_id,
            name="source_query_id",
        )
        frame_start = require_integer(
            self.trajectory_frame_start,
            name="trajectory_frame_start",
        )
        expected_start = self.context.o_minus.frame_stop - 1
        if frame_start != expected_start:
            raise ValueError(
                "trajectory_frame_start must be the factual-prefix endpoint "
                f"{expected_start}"
            )

        source_kind = require_nonempty_string(
            self.source_kind,
            name="source_kind",
        )
        source_revision = require_nonempty_string(
            self.source_revision,
            name="source_revision",
        )
        source_content_sha256 = validate_sha256(
            self.source_content_sha256,
            name="source_content_sha256",
        )
        source_artifact_id = require_optional_string(
            self.source_artifact_id,
            name="source_artifact_id",
        )

        nodes = readonly_integer_array(self.node_indices, name="node_indices")
        if nodes.ndim != 1 or not len(nodes):
            raise ValueError("node_indices must be a nonempty vector")
        if not np.array_equal(nodes, np.arange(len(nodes), dtype=np.int64)):
            raise ValueError(
                "schema v1 requires the canonical zero-based dense-prefix node order"
            )

        raw_positions = np.asarray(self.positions_m)
        if raw_positions.dtype.kind not in {"f", "i", "u"}:
            raise ValueError("positions_m must contain real numeric values")
        positions = raw_positions.astype(np.float64, copy=True)
        if (
            positions.ndim != 3
            or positions.shape[0] < 1
            or positions.shape[1] != len(nodes)
            or positions.shape[2] != 3
        ):
            raise ValueError("positions_m must have shape (T>=1, N, 3)")

        supplied_validity = np.asarray(self.validity_mask)
        if supplied_validity.dtype.kind != "b":
            raise ValueError("validity_mask must contain booleans")
        if supplied_validity.shape == positions.shape:
            valid = np.all(supplied_validity, axis=2)
        elif supplied_validity.shape == positions.shape[:2]:
            valid = supplied_validity.copy()
        else:
            raise ValueError("validity_mask must have shape (T, N) or (T, N, 3)")
        if np.any(valid & ~np.all(np.isfinite(positions), axis=2)):
            raise ValueError("valid target points must contain finite coordinates")
        if not np.any(valid):
            raise ValueError("held-out target must contain at least one valid point")
        positions[~valid] = np.nan
        positions.setflags(write=False)
        valid.setflags(write=False)

        frame_stop = frame_start + len(positions)
        if frame_stop != self.context.u_cf.frame_stop:
            raise ValueError(
                "target trajectory must end at the counterfactual action stop; "
                f"expected {self.context.u_cf.frame_stop}, got {frame_stop}"
            )

        metadata = validated_json_mapping(
            self.metadata,
            error_message="held-out target metadata must be finite JSON data",
        )
        object.__setattr__(self, "source_query_id", source_query_id)
        object.__setattr__(self, "trajectory_frame_start", frame_start)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "validity_mask", valid)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_content_sha256", source_content_sha256)
        object.__setattr__(self, "source_artifact_id", source_artifact_id)
        object.__setattr__(self, "metadata", metadata)

    @property
    def trajectory_frame_stop(self) -> int:
        return self.trajectory_frame_start + len(self.positions_m)

    @property
    def point_count(self) -> int:
        return len(self.node_indices)

    @property
    def source(self) -> dict[str, str | None]:
        return {
            "kind": self.source_kind,
            "revision": self.source_revision,
            "content_sha256": self.source_content_sha256,
            "artifact_id": self.source_artifact_id,
        }

    def payload_hashes(self) -> dict[str, str]:
        return {
            "node_indices_sha256": array_sha256(self.node_indices),
            "positions_m_sha256": array_sha256(self.positions_m),
            "validity_mask_sha256": array_sha256(self.validity_mask),
        }

    def _descriptor_without_id(self) -> dict[str, Any]:
        return {
            "schema": HELD_OUT_PHYSICAL_TARGET_SCHEMA,
            "schema_version": HELD_OUT_PHYSICAL_TARGET_SCHEMA_VERSION,
            "context": self.context.as_dict(),
            "source_query_id": self.source_query_id,
            "trajectory_frame_interval": [
                self.trajectory_frame_start,
                self.trajectory_frame_stop,
            ],
            "node_order": NODE_ORDER_DENSE_PREFIX_V1,
            "source": self.source,
            "metadata": plain_json(self.metadata),
            "payload": self.payload_hashes(),
        }

    @property
    def artifact_id(self) -> str:
        encoded = canonical_json(self._descriptor_without_id()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        descriptor = self._descriptor_without_id()
        descriptor["artifact_id"] = self.artifact_id
        return descriptor

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "protocol_id": self.context.protocol_id,
            "case_id": self.context.case_id,
            "counterfactual_action_id": self.context.u_cf.action_id,
            "source_query_id": self.source_query_id,
            "trajectory_frame_interval": [
                self.trajectory_frame_start,
                self.trajectory_frame_stop,
            ],
            "trajectory_length": len(self.positions_m),
            "point_count": self.point_count,
            "valid_point_frames": int(np.sum(self.validity_mask)),
            "source": self.source,
        }

    def require_compatible_physical_posterior(
        self,
        posterior: PhysicalPosterior,
    ) -> None:
        """Fail before scoring unless target and posterior identify one query."""

        if not isinstance(posterior, PhysicalPosterior):
            raise TypeError("posterior must be a PhysicalPosterior")
        if self.context.as_dict() != posterior.context.as_dict():
            raise ValueError("held-out target causal context does not match posterior")
        if self.source_query_id != posterior.source_query_id:
            raise ValueError("held-out target source_query_id does not match posterior")
        expected_shape = posterior.readout_trajectories_m.shape[1:]
        if self.positions_m.shape != expected_shape:
            raise ValueError(
                "held-out target shape does not match posterior readout trajectory: "
                f"{self.positions_m.shape} != {expected_shape}"
            )
        expected_nodes = np.arange(expected_shape[1], dtype=np.int64)
        if not np.array_equal(self.node_indices, expected_nodes):
            raise ValueError("held-out target node order does not match posterior")


def _save_archive(handle: BinaryIO, target: HeldOutPhysicalTarget) -> None:
    np.savez_compressed(
        handle,
        descriptor_json=np.asarray(canonical_json(target.descriptor())),
        node_indices=target.node_indices,
        positions_m=target.positions_m,
        validity_mask=target.validity_mask,
    )


def save_held_out_physical_target(
    path: str | Path,
    target: HeldOutPhysicalTarget,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically publish a safe held-out target archive."""

    expected_id = target.artifact_id

    def validate(temporary: Path) -> None:
        restored = load_held_out_physical_target(temporary)
        if restored.artifact_id != expected_id:
            raise ValueError("held-out target changed during serialization")

    atomic_write_binary(
        path,
        lambda handle: _save_archive(handle, target),
        overwrite=overwrite,
        validate=validate,
    )


def _parse_descriptor(value: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("descriptor_json must be one scalar string")
    scalar = array.item()
    if isinstance(scalar, bytes):
        try:
            scalar = scalar.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("descriptor_json is not valid UTF-8") from error
    if type(scalar) is not str:
        raise ValueError("descriptor_json must be one scalar string")
    try:
        parsed = json.loads(
            scalar,
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=reject_nonfinite_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("descriptor_json is invalid") from error
    return require_mapping(parsed, name="held-out target descriptor")


def load_held_out_physical_target(path: str | Path) -> HeldOutPhysicalTarget:
    """Load and independently revalidate a held-out target archive."""

    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("held-out target archive contains duplicate entries")
        actual_fields = set(archive.files)
        if actual_fields != _ARCHIVE_FIELDS:
            raise ValueError(
                "held-out target archive fields do not match schema; "
                f"missing={sorted(_ARCHIVE_FIELDS - actual_fields)}, "
                f"unexpected={sorted(actual_fields - _ARCHIVE_FIELDS)}"
            )
        descriptor = validate_target_descriptor(
            _parse_descriptor(np.asarray(archive["descriptor_json"]))
        )
        node_indices = np.asarray(archive["node_indices"])
        positions_m = np.asarray(archive["positions_m"])
        validity_mask = np.asarray(archive["validity_mask"])
        if node_indices.dtype != np.dtype(np.int64):
            raise ValueError("node_indices must use int64 storage")
        if positions_m.dtype != np.dtype(np.float64):
            raise ValueError("positions_m must use float64 storage")
        if validity_mask.dtype != np.dtype(np.bool_):
            raise ValueError("validity_mask must use boolean storage")

    source = descriptor["source"]
    payload = descriptor["payload"]
    interval = descriptor["trajectory_frame_interval"]
    target = HeldOutPhysicalTarget(
        context=CausalContext.from_dict(descriptor["context"]),
        source_query_id=descriptor["source_query_id"],
        trajectory_frame_start=interval[0],
        node_indices=node_indices,
        positions_m=positions_m,
        validity_mask=validity_mask,
        source_kind=source["kind"],
        source_revision=source["revision"],
        source_content_sha256=source["content_sha256"],
        source_artifact_id=source["artifact_id"],
        metadata=require_mapping(descriptor["metadata"], name="metadata"),
    )
    if descriptor["artifact_id"] != target.artifact_id:
        raise ValueError("held-out target artifact_id does not match payload")
    if dict(payload) != target.payload_hashes():
        raise ValueError("held-out target payload hashes do not match arrays")
    if interval[1] != target.trajectory_frame_stop:
        raise ValueError("held-out target frame interval does not match payload")
    return target


__all__ = [
    "HELD_OUT_PHYSICAL_TARGET_SCHEMA",
    "HELD_OUT_PHYSICAL_TARGET_SCHEMA_VERSION",
    "NODE_ORDER_DENSE_PREFIX_V1",
    "HeldOutPhysicalTarget",
    "load_held_out_physical_target",
    "save_held_out_physical_target",
]
