"""Validate and bind portable 4-D observation lineage to Causal4D beliefs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import TwinBelief
from .prob4d_observation_lineage import (
    is_prob4d_causal_observation_descriptor,
    validate_prob4d_causal_observation_metadata,
)

OBSERVATION_BELIEF_SCHEMA = "phys4d.observation_belief"
OBSERVATION_BELIEF_VERSION = 1
_REQUIRED_ARRAYS = {
    "declared_frame_ids",
    "mean_xyz_m",
    "frame_ids",
    "entity_ids",
    "view_indices",
    "window_indices",
    "correlation_group_ids",
    "factor_group_ids",
    "prior_reliability",
    "association_probability",
    "local_covariance_m2",
    "low_rank_factor_m",
    "group_ids",
    "group_prior_nominal_probability",
    "group_composite_weight",
}


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(array.shape, separators=(",", ":")).encode("ascii")
    )
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def compute_observation_artifact_id(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Compute the cross-repository observation contract content address."""

    payload = dict(descriptor)
    payload.pop("artifact_id", None)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(values).encode("ascii"))
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _bounded_probability(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(
        (result < 0.0) | (result > 1.0)
    ):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _validated_json_mapping(
    values: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(
                dict(values),
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error


@dataclass(frozen=True)
class ObservationLineage:
    """Small immutable view of a validated ObservationBeliefV1 archive."""

    artifact_id: str
    case_id: str
    stream_id: str
    causal_frame_stop: int
    minimum_frame_id: int
    maximum_frame_id: int
    observation_count: int
    group_count: int
    factor_rank: int
    source_repository: str
    source_revision: str
    source_artifact_sha256: str
    provider_validation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_sha256(self.artifact_id, name="artifact_id")
        _validate_sha256(
            self.source_artifact_sha256,
            name="source_artifact_sha256",
        )
        if not self.case_id or not self.stream_id:
            raise ValueError("lineage case and stream must be nonempty")
        if not self.source_repository or not self.source_revision:
            raise ValueError("lineage source must be nonempty")
        if self.causal_frame_stop < 1:
            raise ValueError(
                "lineage causal frame stop must be positive"
            )
        if not 0 <= self.minimum_frame_id <= self.maximum_frame_id:
            raise ValueError("lineage frame range is invalid")
        if self.maximum_frame_id >= self.causal_frame_stop:
            raise ValueError(
                "lineage crosses its causal frame boundary"
            )
        if self.observation_count < 1 or self.group_count < 1:
            raise ValueError(
                "lineage must describe observations and groups"
            )
        if self.factor_rank < 0:
            raise ValueError("lineage factor rank must be nonnegative")
        object.__setattr__(
            self,
            "provider_validation",
            _validated_json_mapping(
                self.provider_validation,
                name="provider validation",
            ),
        )

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source_observation_belief_id": self.artifact_id,
            "source_observation_schema": OBSERVATION_BELIEF_SCHEMA,
            "source_observation_schema_version": (
                OBSERVATION_BELIEF_VERSION
            ),
            "source_observation_case_id": self.case_id,
            "source_observation_stream_id": self.stream_id,
            "source_observation_causal_frame_stop": (
                self.causal_frame_stop
            ),
            "source_observation_repository": self.source_repository,
            "source_observation_revision": self.source_revision,
            "source_observation_artifact_sha256": (
                self.source_artifact_sha256
            ),
        }
        if self.provider_validation:
            metadata["source_observation_provider_validation"] = dict(
                self.provider_validation
            )
        return metadata


def load_observation_lineage(path: str | Path) -> ObservationLineage:
    """Validate an ObservationBeliefV1 archive without importing its provider."""

    with np.load(path, allow_pickle=False) as archive:
        if "descriptor_json" not in archive:
            raise ValueError(
                "observation artifact has no descriptor_json"
            )
        descriptor = json.loads(str(archive["descriptor_json"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    if descriptor.get("schema_name") != OBSERVATION_BELIEF_SCHEMA:
        raise ValueError("unsupported observation-belief schema")
    if (
        int(descriptor.get("schema_version", -1))
        != OBSERVATION_BELIEF_VERSION
    ):
        raise ValueError("unsupported observation-belief version")
    missing = _REQUIRED_ARRAYS - arrays.keys()
    extra = arrays.keys() - _REQUIRED_ARRAYS
    if missing or extra:
        raise ValueError(
            "observation artifact arrays changed; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    expected = str(descriptor.get("artifact_id", ""))
    _validate_sha256(expected, name="artifact_id")
    actual = compute_observation_artifact_id(descriptor, arrays)
    if actual != expected:
        raise ValueError(
            "observation artifact digest does not match its payload"
        )

    view_names = tuple(
        map(str, descriptor.get("view_names", ()))
    )
    window_names = tuple(
        map(str, descriptor.get("window_names", ()))
    )
    factor_names = tuple(
        map(str, descriptor.get("factor_names", ()))
    )
    if (
        not view_names
        or any(not name for name in view_names)
        or not window_names
        or any(not name for name in window_names)
        or any(not name for name in factor_names)
    ):
        raise ValueError(
            "observation view, window, or factor names are invalid"
        )
    _validate_sha256(
        str(descriptor.get("source_artifact_sha256", "")),
        name="source_artifact_sha256",
    )

    declared_frames = np.asarray(
        arrays["declared_frame_ids"],
        dtype=np.int64,
    )
    mean = np.asarray(arrays["mean_xyz_m"], dtype=np.float64)
    frame_ids = np.asarray(arrays["frame_ids"], dtype=np.int64)
    entity_ids = np.asarray(arrays["entity_ids"], dtype=np.int64)
    view_indices = np.asarray(
        arrays["view_indices"],
        dtype=np.int64,
    )
    window_indices = np.asarray(
        arrays["window_indices"],
        dtype=np.int64,
    )
    correlation_groups = np.asarray(
        arrays["correlation_group_ids"],
        dtype=np.int64,
    )
    factor_groups = np.asarray(
        arrays["factor_group_ids"],
        dtype=np.int64,
    )
    local_covariance = np.asarray(
        arrays["local_covariance_m2"],
        dtype=np.float64,
    )
    factors = np.asarray(
        arrays["low_rank_factor_m"],
        dtype=np.float64,
    )
    group_ids = np.asarray(arrays["group_ids"], dtype=np.int64)
    group_prior = _bounded_probability(
        arrays["group_prior_nominal_probability"],
        name="group prior nominal probability",
    )
    group_weight = np.asarray(
        arrays["group_composite_weight"],
        dtype=np.float64,
    )
    prior_reliability = _bounded_probability(
        arrays["prior_reliability"],
        name="prior reliability",
    )
    association = _bounded_probability(
        arrays["association_probability"],
        name="association probability",
    )
    causal_stop = int(descriptor["causal_frame_stop"])
    if (
        declared_frames.ndim != 1
        or len(declared_frames) == 0
        or np.any(np.diff(declared_frames) <= 0)
        or np.any(declared_frames < 0)
        or np.any(declared_frames >= causal_stop)
    ):
        raise ValueError(
            "declared observation frames violate the causal boundary"
        )
    if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) == 0:
        raise ValueError("observation means must have shape (N, 3)")
    count = len(mean)
    for name, values in (
        ("frame_ids", frame_ids),
        ("entity_ids", entity_ids),
        ("view_indices", view_indices),
        ("window_indices", window_indices),
        ("correlation_group_ids", correlation_groups),
        ("factor_group_ids", factor_groups),
        ("prior_reliability", prior_reliability),
        ("association_probability", association),
    ):
        if values.shape != (count,):
            raise ValueError(
                f"{name} must identify every observation row"
            )
    if not np.all(np.isin(frame_ids, declared_frames)):
        raise ValueError(
            "observation frame identities are inconsistent"
        )
    if np.any(entity_ids < 0):
        raise ValueError(
            "observation entity identities must be nonnegative"
        )
    if np.any(view_indices < 0) or np.any(
        view_indices >= len(view_names)
    ):
        raise ValueError("observation view indices are invalid")
    if np.any(window_indices < 0) or np.any(
        window_indices >= len(window_names)
    ):
        raise ValueError("observation window indices are invalid")
    if np.any(correlation_groups < 0) or np.any(factor_groups < 0):
        raise ValueError(
            "observation group identities must be nonnegative"
        )
    if local_covariance.shape != (count, 3, 3):
        raise ValueError(
            "observation local covariance shape changed"
        )
    symmetric = 0.5 * (
        local_covariance
        + np.swapaxes(local_covariance, 1, 2)
    )
    if (
        not np.all(np.isfinite(local_covariance))
        or not np.allclose(
            local_covariance,
            symmetric,
            atol=1e-12,
            rtol=1e-10,
        )
        or np.any(
            np.min(np.linalg.eigvalsh(symmetric), axis=1) <= 0.0
        )
    ):
        raise ValueError(
            "observation local covariance is not positive definite"
        )
    if factors.shape != (count, 3, len(factor_names)) or not np.all(
        np.isfinite(factors)
    ):
        raise ValueError(
            "observation low-rank factor shape changed"
        )
    if not np.array_equal(
        group_ids,
        np.unique(correlation_groups),
    ):
        raise ValueError(
            "observation group IDs do not match row assignments"
        )
    if (
        group_prior.shape != group_ids.shape
        or group_weight.shape != group_ids.shape
    ):
        raise ValueError(
            "observation group metadata shape changed"
        )
    if not np.all(np.isfinite(group_weight)) or np.any(
        (group_weight <= 0.0) | (group_weight > 1.0)
    ):
        raise ValueError(
            "observation group weights must lie in (0, 1]"
        )
    if not np.all(np.isfinite(mean)):
        raise ValueError("observation means must be finite")

    order = np.lexsort(
        (
            window_indices,
            view_indices,
            entity_ids,
            frame_ids,
        )
    )
    keys = np.column_stack(
        (
            frame_ids[order],
            entity_ids[order],
            view_indices[order],
            window_indices[order],
        )
    )
    if len(keys) > 1 and np.any(
        np.all(keys[1:] == keys[:-1], axis=1)
    ):
        raise ValueError("observation row identity is not unique")

    provider_validation: dict[str, object] = {}
    if is_prob4d_causal_observation_descriptor(descriptor):
        provider_validation = (
            validate_prob4d_causal_observation_metadata(
                descriptor,
                arrays,
            )
        )

    return ObservationLineage(
        artifact_id=actual,
        case_id=str(descriptor["case_id"]),
        stream_id=str(descriptor["stream_id"]),
        causal_frame_stop=causal_stop,
        minimum_frame_id=int(declared_frames[0]),
        maximum_frame_id=int(declared_frames[-1]),
        observation_count=count,
        group_count=len(group_ids),
        factor_rank=factors.shape[2],
        source_repository=str(descriptor["source_repository"]),
        source_revision=str(descriptor["source_revision"]),
        source_artifact_sha256=str(
            descriptor["source_artifact_sha256"]
        ),
        provider_validation=provider_validation,
    )


def validate_twin_belief_observation_lineage(
    twin_belief: TwinBelief,
    lineage: ObservationLineage,
    *,
    require_bound: bool = True,
) -> dict[str, Any]:
    """Check case, O-minus containment, and content-addressed binding."""

    if twin_belief.context.case_id != lineage.case_id:
        raise ValueError(
            "observation and twin belief identify different cases"
        )
    if (
        lineage.minimum_frame_id
        < twin_belief.context.o_minus.frame_start
    ):
        raise ValueError(
            "observation begins before the TwinBelief O- boundary"
        )
    if (
        lineage.causal_frame_stop
        > twin_belief.context.o_minus.frame_stop
    ):
        raise ValueError(
            "observation uses frames beyond the TwinBelief O- boundary"
        )
    bound_id = twin_belief.metadata.get(
        "source_observation_belief_id"
    )
    if bound_id is not None and bound_id != lineage.artifact_id:
        raise ValueError(
            "TwinBelief is bound to a different observation artifact"
        )
    if require_bound and bound_id is None:
        raise ValueError(
            "TwinBelief has no source observation binding"
        )
    return {
        "status": "valid",
        "twin_belief_id": twin_belief.artifact_id,
        "observation_belief_id": lineage.artifact_id,
        "case_id": lineage.case_id,
        "lineage_bound": bound_id == lineage.artifact_id,
        "observation_frame_range": [
            lineage.minimum_frame_id,
            lineage.maximum_frame_id,
        ],
        "observation_causal_frame_stop": (
            lineage.causal_frame_stop
        ),
        "twin_o_minus_frame_range": [
            twin_belief.context.o_minus.frame_start,
            twin_belief.context.o_minus.frame_stop,
        ],
        "observation_count": lineage.observation_count,
        "group_count": lineage.group_count,
        "factor_rank": lineage.factor_rank,
        "provider_validation": dict(lineage.provider_validation),
    }


def bind_twin_belief_observation_lineage(
    twin_belief: TwinBelief,
    lineage: ObservationLineage,
) -> TwinBelief:
    """Return a new content-addressed TwinBelief with explicit source lineage.

    This operation should be called only by the estimator that actually consumed
    the observation artifact. It does not claim that merely validating an unused
    observation made it part of the belief.
    """

    validate_twin_belief_observation_lineage(
        twin_belief,
        lineage,
        require_bound=False,
    )
    metadata = dict(twin_belief.metadata)
    existing = metadata.get("source_observation_belief_id")
    if existing is not None and existing != lineage.artifact_id:
        raise ValueError(
            "TwinBelief already has incompatible observation lineage"
        )
    metadata.update(lineage.metadata())
    return replace(twin_belief, metadata=metadata)


__all__ = [
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "ObservationLineage",
    "array_sha256",
    "bind_twin_belief_observation_lineage",
    "compute_observation_artifact_id",
    "load_observation_lineage",
    "validate_twin_belief_observation_lineage",
]
