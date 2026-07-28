"""Typed low-rank graph-discrepancy beliefs with provenance-complete storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.immutable_json import validated_json_mapping

from causal4d.contracts import array_sha256
from causal4d.observation_evidence import GroupedObservationEvidence


GRAPH_DISCREPANCY_BELIEF_VERSION = 1


def _readonly(values: np.ndarray, *, dtype: type | None = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        _validate_sha256(self.basis_sha256, name="basis_sha256")
        if self.source_physical_posterior_id is not None:
            _validate_sha256(
                self.source_physical_posterior_id,
                name="source_physical_posterior_id",
            )
        if not self.transition_model_id or not self.innovation_model_id:
            raise ValueError("discrepancy model identifiers must be nonempty")
        if not self.component_ids or len(set(self.component_ids)) != len(
            self.component_ids
        ):
            raise ValueError("component_ids must be nonempty and unique")

        mean = _readonly(self.coefficient_mean_m)
        covariance = _readonly(self.coefficient_covariance_m2)
        projection = _readonly(self.projection_variance_m2)
        component_count = len(self.component_ids)
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
            np.all(np.isfinite(value))
            for value in (mean, covariance, projection)
        ):
            raise ValueError("discrepancy belief arrays must be finite")
        if np.any(projection < 0.0):
            raise ValueError("projection variance must be nonnegative")
        _positive_semidefinite(covariance, name="coefficient covariance")

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
            "metadata": self.metadata,
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
    """Write a JSON manifest and non-pickled NPZ payload."""

    manifest = Path(manifest_path)
    payload = manifest.with_suffix(".npz")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        payload,
        coefficient_mean_m=belief.coefficient_mean_m,
        coefficient_covariance_m2=belief.coefficient_covariance_m2,
        projection_variance_m2=belief.projection_variance_m2,
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
        "metadata": belief.metadata,
        "payload": {
            "path": payload.name,
            "sha256": _file_sha256(payload),
        },
    }
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


def load_graph_discrepancy_belief(
    manifest_path: str | Path,
) -> GraphDiscrepancyBelief:
    """Load and verify a graph-discrepancy belief artifact."""

    manifest = Path(manifest_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("artifact_kind") != "GraphDiscrepancyBelief":
        raise ValueError("manifest is not a graph-discrepancy belief")
    if int(record.get("version", -1)) != GRAPH_DISCREPANCY_BELIEF_VERSION:
        raise ValueError("unsupported graph-discrepancy belief version")
    payload = manifest.parent / str(record["payload"]["path"])
    if _file_sha256(payload) != record["payload"]["sha256"]:
        raise ValueError("graph-discrepancy payload checksum changed")
    with np.load(payload, allow_pickle=False) as arrays:
        belief = GraphDiscrepancyBelief(
            basis_sha256=str(record["basis_sha256"]),
            component_ids=tuple(map(str, record["component_ids"])),
            coefficient_mean_m=arrays["coefficient_mean_m"],
            coefficient_covariance_m2=arrays["coefficient_covariance_m2"],
            projection_variance_m2=arrays["projection_variance_m2"],
            transition_model_id=str(record["transition_model_id"]),
            innovation_model_id=str(record["innovation_model_id"]),
            source_physical_posterior_id=record.get("source_physical_posterior_id"),
            metadata=record.get("metadata", {}),
        )
    if belief.artifact_id != record["artifact_id"]:
        raise ValueError("graph-discrepancy artifact identifier changed")
    return belief
