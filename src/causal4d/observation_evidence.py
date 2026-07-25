"""Typed grouped observation evidence for correlation-aware Causal4D updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _json_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    import json

    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain finite JSON data") from error


@dataclass(frozen=True)
class ObservationGroup:
    """One jointly scored observation group.

    A group contains scalar coordinates selected from a dense rollout by parallel
    frame, node, and coordinate index vectors. ``covariance_m2`` is the covariance
    of the complete selected vector, not a per-coordinate variance shortcut.
    ``prior_nominal_probability`` is fixed before evaluating the residual.
    """

    group_id: str
    values_m: np.ndarray
    frame_indices: np.ndarray
    node_indices: np.ndarray
    coordinate_indices: np.ndarray
    covariance_m2: np.ndarray
    contributor_ids: tuple[str, ...]
    prior_nominal_probability: float = 0.95
    outlier_scale_multiplier: float = 100.0
    degrees_of_freedom: float = 4.0
    composite_weight: float = 1.0
    source_id: str = "unknown"
    view_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id or not self.source_id:
            raise ValueError("group_id and source_id must be nonempty")
        values = _readonly(self.values_m)
        frames = _readonly(self.frame_indices, dtype=np.int64)
        nodes = _readonly(self.node_indices, dtype=np.int64)
        coordinates = _readonly(self.coordinate_indices, dtype=np.int64)
        covariance = _readonly(self.covariance_m2)
        count = len(values)
        if values.ndim != 1 or count == 0:
            raise ValueError("values_m must be a nonempty vector")
        if frames.shape != (count,) or nodes.shape != (count,) or coordinates.shape != (count,):
            raise ValueError("frame, node, and coordinate indices must match values_m")
        if np.any(frames < 0) or np.any(nodes < 0) or np.any(coordinates < 0):
            raise ValueError("observation indices must be nonnegative")
        if not np.all(np.isfinite(values)):
            raise ValueError("observation values must be finite")
        if covariance.shape != (count, count):
            raise ValueError("covariance_m2 must have shape (coordinate, coordinate)")
        if not np.all(np.isfinite(covariance)) or not np.allclose(
            covariance, covariance.T, atol=1e-12, rtol=1e-10
        ):
            raise ValueError("covariance_m2 must be finite and symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if np.min(eigenvalues) <= 0.0:
            raise ValueError("covariance_m2 must be positive definite")
        if not 0.0 < self.prior_nominal_probability < 1.0:
            raise ValueError("prior_nominal_probability must lie in (0, 1)")
        if self.outlier_scale_multiplier <= 1.0:
            raise ValueError("outlier_scale_multiplier must exceed one")
        if self.degrees_of_freedom <= 2.0:
            raise ValueError("degrees_of_freedom must exceed two")
        if not 0.0 < self.composite_weight <= 1.0:
            raise ValueError("composite_weight must lie in (0, 1]")
        if not self.contributor_ids or any(not value for value in self.contributor_ids):
            raise ValueError("contributor_ids must contain nonempty identities")
        if len(set(self.contributor_ids)) != len(self.contributor_ids):
            raise ValueError("contributor_ids must be unique within a group")
        object.__setattr__(self, "values_m", values)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "coordinate_indices", coordinates)
        object.__setattr__(self, "covariance_m2", covariance)
        object.__setattr__(self, "contributor_ids", tuple(self.contributor_ids))
        object.__setattr__(self, "metadata", _json_metadata(self.metadata))

    @property
    def coordinate_count(self) -> int:
        return len(self.values_m)

    def validate_rollout_shape(self, shape: Sequence[int]) -> None:
        if len(shape) != 3:
            raise ValueError("rollout shape must be (frame, node, coordinate)")
        frame_count, node_count, coordinate_count = map(int, shape)
        if np.any(self.frame_indices >= frame_count):
            raise ValueError(f"group {self.group_id!r} references an unavailable frame")
        if np.any(self.node_indices >= node_count):
            raise ValueError(f"group {self.group_id!r} references an unavailable node")
        if np.any(self.coordinate_indices >= coordinate_count):
            raise ValueError(f"group {self.group_id!r} references an unavailable coordinate")

    def selected_predictions(self, trajectories_m: np.ndarray) -> np.ndarray:
        trajectories = np.asarray(trajectories_m, dtype=float)
        if trajectories.ndim < 3:
            raise ValueError("trajectories_m must end in (frame, node, coordinate)")
        self.validate_rollout_shape(trajectories.shape[-3:])
        return trajectories[
            ...,
            self.frame_indices,
            self.node_indices,
            self.coordinate_indices,
        ]


@dataclass(frozen=True)
class GroupedObservationEvidence:
    """A collection of observation groups with contributor-aware power caps.

    Reusing a contributor in multiple groups automatically divides each affected
    group power by that contributor's multiplicity. Exact duplicated evidence with
    the same contributor identity therefore cannot sharpen the posterior.
    """

    groups: tuple[ObservationGroup, ...]
    evidence_id: str = "grouped_observation_evidence"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not groups or not self.evidence_id:
            raise ValueError("grouped evidence must have an identity and at least one group")
        identifiers = [group.group_id for group in groups]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("observation group IDs must be unique")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "metadata", _json_metadata(self.metadata))

    @property
    def contributor_multiplicity(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for group in self.groups:
            for contributor in group.contributor_ids:
                result[contributor] = result.get(contributor, 0) + 1
        return result

    @property
    def effective_group_weights(self) -> tuple[float, ...]:
        multiplicity = self.contributor_multiplicity
        return tuple(
            group.composite_weight
            / max(multiplicity[contributor] for contributor in group.contributor_ids)
            for group in self.groups
        )

    def validate_prefix(self, *, prefix_frame_count: int, rollout_shape: Sequence[int]) -> None:
        if prefix_frame_count < 2:
            raise ValueError("prefix_frame_count must reveal at least one O-plus frame")
        for group in self.groups:
            group.validate_rollout_shape(rollout_shape)
            if np.any(group.frame_indices <= 0):
                raise ValueError("grouped O-plus evidence may not reuse the endpoint frame")
            if np.any(group.frame_indices >= prefix_frame_count):
                raise ValueError("grouped evidence crosses the declared O-plus prefix")

    @classmethod
    def from_dense_prefix(
        cls,
        observations_m: np.ndarray,
        *,
        prefix_frame_count: int,
        scale_m: float,
        mask: np.ndarray | None = None,
        prior_nominal_probability: float = 0.95,
        outlier_scale_multiplier: float = 100.0,
        degrees_of_freedom: float = 4.0,
        source_id: str = "dense_observation",
    ) -> "GroupedObservationEvidence":
        """Build one covariance-aware group per O-plus frame.

        This convenience constructor preserves frame-level correlation boundaries.
        More detailed feeders should construct groups directly and supply their full
        metric covariance and contributor provenance.
        """

        observations = np.asarray(observations_m, dtype=float)
        if observations.ndim != 3 or observations.shape[2] not in {2, 3}:
            raise ValueError("observations_m must have shape (frame, node, 2|3)")
        if not 2 <= prefix_frame_count <= len(observations):
            raise ValueError("prefix_frame_count must reveal O-plus frames")
        if scale_m <= 0.0 or not np.isfinite(scale_m):
            raise ValueError("scale_m must be positive and finite")
        valid = np.all(np.isfinite(observations), axis=2)
        if mask is not None:
            supplied = np.asarray(mask, dtype=bool)
            if supplied.shape == observations.shape:
                supplied = np.all(supplied, axis=2)
            if supplied.shape != observations.shape[:2]:
                raise ValueError("mask must have shape (T, N) or (T, N, C)")
            valid &= supplied
        groups = []
        coordinate_count = observations.shape[2]
        for frame in range(1, prefix_frame_count):
            nodes = np.flatnonzero(valid[frame])
            if len(nodes) == 0:
                continue
            frame_indices = np.repeat(frame, len(nodes) * coordinate_count)
            node_indices = np.repeat(nodes, coordinate_count)
            coordinate_indices = np.tile(np.arange(coordinate_count), len(nodes))
            values = observations[frame, nodes].reshape(-1)
            covariance = np.eye(len(values), dtype=float) * scale_m**2
            groups.append(
                ObservationGroup(
                    group_id=f"{source_id}:frame:{frame}",
                    values_m=values,
                    frame_indices=frame_indices,
                    node_indices=node_indices,
                    coordinate_indices=coordinate_indices,
                    covariance_m2=covariance,
                    contributor_ids=(f"{source_id}:frame:{frame}",),
                    prior_nominal_probability=prior_nominal_probability,
                    outlier_scale_multiplier=outlier_scale_multiplier,
                    degrees_of_freedom=degrees_of_freedom,
                    source_id=source_id,
                )
            )
        if not groups:
            raise ValueError("dense prefix contains no valid O-plus observations")
        return cls(
            groups=tuple(groups),
            evidence_id=f"{source_id}:prefix:{prefix_frame_count}",
            metadata={
                "constructor": "from_dense_prefix",
                "prefix_frame_count_including_endpoint": prefix_frame_count,
            },
        )
