"""Typed low-rank graph-discrepancy beliefs with provenance-complete storage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from causal4d.artifact_io import (
    load_npz_bytes,
    load_strict_json_object,
    read_regular_file,
    read_regular_file_beneath,
)
from causal4d.atomic_io import atomic_write_binary, atomic_write_json
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.observation_evidence import GroupedObservationEvidence


GRAPH_DISCREPANCY_BELIEF_VERSION = 1

_MANIFEST_FIELDS = frozenset(
    {
        "artifact_kind",
        "version",
        "artifact_id",
        "basis_sha256",
        "component_ids",
        "transition_model_id",
        "innovation_model_id",
        "source_physical_posterior_id",
        "metadata",
        "payload",
    }
)
_PAYLOAD_FIELDS = frozenset({"path", "sha256"})
_PAYLOAD_ARRAYS = frozenset(
    {
        "coefficient_mean_m",
        "coefficient_covariance_m2",
        "projection_variance_m2",
    }
)


def _readonly(values: np.ndarray, *, dtype: type | None = float) -> np.ndarray:
    return readonly_array(values, dtype=dtype)


def _validated_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _validated_component_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("component_ids must be a nonempty JSON array")
    result = tuple(
        _nonempty_string(identifier, name=f"component_ids[{index}]")
        for index, identifier in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError("component_ids must be unique")
    return result


def _require_float64_array(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use float64")
    return result


def _positive_semidefinite(covariance: np.ndarray, *, name: str) -> None:
    if not np.allclose(
        covariance,
        covariance.swapaxes(-1, -2),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be symmetric")
    minimum = float(np.min(np.linalg.eigvalsh(covariance), initial=0.0))
    if minimum < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")


@dataclass(frozen=True)
class GraphDiscrepancyBelief:
    """Component-wise Gaussian belief over graph discrepancy coefficients.

    The coefficient state remains separate from simulator state. The graph basis
    is identified by hash so a belief cannot silently move between topologies or
    basis conventions.
    """

    basis_sha256: str
    component_ids: tuple[str, ...]
    coefficient_mean_m: np.ndarray
    coefficient_covariance_m2: np.ndarray
    projection_variance_m2: np.ndarray
    transition_model_id: str
    innovation_model_id: str
    source_physical_posterior_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validated_sha256(self.basis_sha256, name="basis_sha256")
        if self.source_physical_posterior_id is not None:
            _validated_sha256(
                self.source_physical_posterior_id,
                name="source_physical_posterior_id",
            )
        _nonempty_string(self.transition_model_id, name="transition_model_id")
        _nonempty_string(self.innovation_model_id, name="innovation_model_id")
        if isinstance(self.component_ids, (str, bytes)):
            raise ValueError("component_ids must be a nonempty sequence")
        component_ids = tuple(self.component_ids)
        if not component_ids or any(
            type(identifier) is not str or not identifier
            for identifier in component_ids
        ):
            raise ValueError("component_ids must contain nonempty strings")
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("component_ids must be unique")

        mean = _readonly(self.coefficient_mean_m)
        covariance = _readonly(self.coefficient_covariance_m2)
        projection = _readonly(self.projection_variance_m2)
        component_count = len(component_ids)
        if mean.ndim != 3 or mean.shape[0] != component_count or mean.shape[2] != 3:
            raise ValueError("coefficient_mean_m must have shape (K, rank, 3)")
        rank = mean.shape[1]
        if rank < 1 or covariance.shape != (component_count, 3, rank, rank):
            raise ValueError(
                "coefficient_covariance_m2 must have shape (K, 3, rank, rank)"
            )
        if projection.shape != (3,):
            raise ValueError("projection_variance_m2 must have shape (3,)")
        if not all(
            np.all(np.isfinite(value)) for value in (mean, covariance, projection)
        ):
            raise ValueError("discrepancy belief arrays must be finite")
        if np.any(projection < 0.0):
            raise ValueError("projection variance must be nonnegative")
        _positive_semidefinite(covariance, name="coefficient covariance")

        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "coefficient_mean_m", mean)
        object.__setattr__(self, "coefficient_covariance_m2", covariance)
        object.__setattr__(self, "projection_variance_m2", projection)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="metadata must contain finite JSON values",
            ),
        )

    @property
    def rank(self) -> int:
        return int(self.coefficient_mean_m.shape[1])

    @property
    def artifact_id(self) -> str:
        descriptor = {
            "artifact_kind": "GraphDiscrepancyBelief",
            "version": GRAPH_DISCREPANCY_BELIEF_VERSION,
            "basis_sha256": self.basis_sha256,
            "component_ids": list(self.component_ids),
            "transition_model_id": self.transition_model_id,
            "innovation_model_id": self.innovation_model_id,
            "source_physical_posterior_id": self.source_physical_posterior_id,
            "metadata": plain_json(self.metadata),
        }
        digest = hashlib.sha256(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in (
            ("coefficient_mean_m", self.coefficient_mean_m),
            ("coefficient_covariance_m2", self.coefficient_covariance_m2),
            ("projection_variance_m2", self.projection_variance_m2),
        ):
            digest.update(name.encode("utf-8"))
            digest.update(array_sha256(values).encode("ascii"))
        return digest.hexdigest()


def graph_discrepancy_group_covariances(
    belief: GraphDiscrepancyBelief,
    graph_basis: np.ndarray,
    evidence: GroupedObservationEvidence,
    *,
    component_ids: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Map coefficient covariance to complete grouped observation covariance.

    The coefficient state is persistent over the scored prefix. Consequently,
    observations of the same graph mode at different frames remain correlated.
    Cross-coordinate covariance is zero because the current belief stores one
    coefficient covariance per Cartesian coordinate. ``projection_variance_m2``
    is retained as an independent coordinate-wise diagonal remainder.

    When ``component_ids`` are supplied, output covariances are selected and
    reordered to match that component order. A particle-level covariance with
    shape ``(P, d, d)`` can then broadcast over rollout components with leading
    shape ``(H, P)`` in the grouped likelihood.
    """

    basis = np.asarray(graph_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[1] != belief.rank:
        raise ValueError("graph_basis must have shape (node_count, belief.rank)")
    if not np.all(np.isfinite(basis)):
        raise ValueError("graph_basis must be finite")
    if array_sha256(basis) != belief.basis_sha256:
        raise ValueError("graph basis hash differs from the discrepancy belief")

    if component_ids is None:
        selected_indices = np.arange(len(belief.component_ids), dtype=np.int64)
    else:
        requested = tuple(component_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("component_ids must be nonempty and unique")
        lookup = {
            identifier: index for index, identifier in enumerate(belief.component_ids)
        }
        missing = [identifier for identifier in requested if identifier not in lookup]
        if missing:
            raise ValueError(f"unknown discrepancy components: {missing}")
        selected_indices = np.asarray(
            [lookup[identifier] for identifier in requested],
            dtype=np.int64,
        )

    coefficient_covariance = belief.coefficient_covariance_m2[selected_indices]
    result: dict[str, np.ndarray] = {}
    for group in evidence.groups:
        if np.any(group.node_indices >= basis.shape[0]):
            raise ValueError(
                f"group {group.group_id!r} references unavailable graph nodes"
            )
        if np.any(group.coordinate_indices >= 3):
            raise ValueError(
                f"group {group.group_id!r} references unavailable coordinates"
            )
        count = group.coordinate_count
        covariance = np.zeros((len(selected_indices), count, count), dtype=float)
        for coordinate in range(3):
            selected = np.flatnonzero(group.coordinate_indices == coordinate)
            if not len(selected):
                continue
            design = basis[group.node_indices[selected]]
            block = np.einsum(
                "ir,krs,js->kij",
                design,
                coefficient_covariance[:, coordinate],
                design,
            )
            covariance[:, selected[:, None], selected[None, :]] = block
        diagonal = np.arange(count)
        covariance[:, diagonal, diagonal] += belief.projection_variance_m2[
            group.coordinate_indices
        ]
        result[group.group_id] = covariance
    return result


def write_graph_discrepancy_belief(
    manifest_path: str | Path,
    belief: GraphDiscrepancyBelief,
) -> dict[str, Any]:
    """Atomically write a strict JSON manifest and non-pickled NPZ payload."""

    manifest = Path(manifest_path)
    payload = manifest.with_suffix(".npz")
    if payload == manifest:
        raise ValueError("graph-discrepancy manifest must not use the .npz suffix")

    def write_payload(handle: BinaryIO) -> None:
        np.savez_compressed(
            handle,
            coefficient_mean_m=belief.coefficient_mean_m,
            coefficient_covariance_m2=belief.coefficient_covariance_m2,
            projection_variance_m2=belief.projection_variance_m2,
        )

    atomic_write_binary(payload, write_payload)
    payload_snapshot = read_regular_file(
        payload,
        name="graph-discrepancy payload",
    )
    record = {
        "artifact_kind": "GraphDiscrepancyBelief",
        "version": GRAPH_DISCREPANCY_BELIEF_VERSION,
        "artifact_id": belief.artifact_id,
        "basis_sha256": belief.basis_sha256,
        "component_ids": list(belief.component_ids),
        "transition_model_id": belief.transition_model_id,
        "innovation_model_id": belief.innovation_model_id,
        "source_physical_posterior_id": belief.source_physical_posterior_id,
        "metadata": plain_json(belief.metadata),
        "payload": {
            "path": payload.name,
            "sha256": payload_snapshot.sha256,
        },
    }
    atomic_write_json(manifest, record)
    return record


def load_graph_discrepancy_belief(
    manifest_path: str | Path,
) -> GraphDiscrepancyBelief:
    """Load and verify one graph-discrepancy belief from exact file bytes."""

    manifest = Path(manifest_path)
    manifest_snapshot = read_regular_file(
        manifest,
        name="graph-discrepancy manifest",
    )
    record = load_strict_json_object(
        manifest_snapshot.payload,
        name="graph-discrepancy manifest",
    )
    _require_exact_fields(record, _MANIFEST_FIELDS, name="graph-discrepancy manifest")

    if record["artifact_kind"] != "GraphDiscrepancyBelief":
        raise ValueError("manifest is not a graph-discrepancy belief")
    if type(record["version"]) is not int:
        raise ValueError("graph-discrepancy version must be an integer")
    if record["version"] != GRAPH_DISCREPANCY_BELIEF_VERSION:
        raise ValueError("unsupported graph-discrepancy belief version")

    artifact_id = _validated_sha256(record["artifact_id"], name="artifact_id")
    basis_sha256 = _validated_sha256(
        record["basis_sha256"],
        name="basis_sha256",
    )
    component_ids = _validated_component_ids(record["component_ids"])
    transition_model_id = _nonempty_string(
        record["transition_model_id"],
        name="transition_model_id",
    )
    innovation_model_id = _nonempty_string(
        record["innovation_model_id"],
        name="innovation_model_id",
    )
    source_physical_posterior_id = record["source_physical_posterior_id"]
    if source_physical_posterior_id is not None:
        source_physical_posterior_id = _validated_sha256(
            source_physical_posterior_id,
            name="source_physical_posterior_id",
        )
    metadata = _require_mapping(record["metadata"], name="metadata")

    payload_record = _require_mapping(record["payload"], name="payload")
    _require_exact_fields(payload_record, _PAYLOAD_FIELDS, name="payload")
    declared_payload_sha256 = _validated_sha256(
        payload_record["sha256"],
        name="payload.sha256",
    )
    payload_snapshot = read_regular_file_beneath(
        manifest.parent,
        payload_record["path"],
        name="graph-discrepancy payload",
    )
    if payload_snapshot.sha256 != declared_payload_sha256:
        raise ValueError("graph-discrepancy payload checksum changed")

    arrays = load_npz_bytes(
        payload_snapshot.payload,
        name="graph-discrepancy payload",
        expected_arrays=_PAYLOAD_ARRAYS,
    )
    coefficient_mean_m = _require_float64_array(
        arrays["coefficient_mean_m"],
        name="coefficient_mean_m",
    )
    coefficient_covariance_m2 = _require_float64_array(
        arrays["coefficient_covariance_m2"],
        name="coefficient_covariance_m2",
    )
    projection_variance_m2 = _require_float64_array(
        arrays["projection_variance_m2"],
        name="projection_variance_m2",
    )

    belief = GraphDiscrepancyBelief(
        basis_sha256=basis_sha256,
        component_ids=component_ids,
        coefficient_mean_m=coefficient_mean_m,
        coefficient_covariance_m2=coefficient_covariance_m2,
        projection_variance_m2=projection_variance_m2,
        transition_model_id=transition_model_id,
        innovation_model_id=innovation_model_id,
        source_physical_posterior_id=source_physical_posterior_id,
        metadata=metadata,
    )
    if belief.artifact_id != artifact_id:
        raise ValueError("graph-discrepancy artifact identifier changed")
    return belief
