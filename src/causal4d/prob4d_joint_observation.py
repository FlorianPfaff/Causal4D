"""Adapt validated Prob4D observation beliefs to full-joint inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.joint_observation import LinearJointObservationEvidence
from causal4d.prob4d_observation_lineage import (
    validate_prob4d_causal_observation_metadata,
)


PROB4D_JOINT_ADAPTER_SCHEMA_VERSION = 1
Prob4DReliabilityPolicy = Literal["require_neutral", "record_only"]


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _required_array(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"Prob4D observation is missing array {name!r}")
    return np.asarray(arrays[name])


def _integer_vector(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    length: int,
) -> np.ndarray:
    values = _required_array(arrays, name)
    if values.dtype.kind not in "iu" or values.shape != (length,):
        raise ValueError(f"Prob4D {name} must be an integer vector of length {length}")
    result = np.asarray(values, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError(f"Prob4D {name} must be nonnegative")
    return result


def _probability_array(
    source: np.ndarray,
    name: str,
    *,
    length: int | None = None,
) -> np.ndarray:
    values = np.asarray(source, dtype=float)
    if values.ndim != 1 or (length is not None and values.shape != (length,)):
        expected = "a vector" if length is None else f"a vector of length {length}"
        raise ValueError(f"Prob4D {name} must be {expected}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"Prob4D {name} must contain probabilities")
    return values


def _normalized_rollout_frames(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(
        _integer(value, name=f"rollout_frame_ids[{index}]")
        for index, value in enumerate(values)
    )
    if not result or len(set(result)) != len(result):
        raise ValueError("rollout_frame_ids must be nonempty and unique")
    return result


def _normalized_entity_mapping(values: Mapping[int, int]) -> dict[int, int]:
    result: dict[int, int] = {}
    for raw_entity, raw_node in values.items():
        entity = _integer(raw_entity, name="entity_to_node key")
        node = _integer(raw_node, name=f"entity_to_node[{entity}]")
        if entity in result:
            raise ValueError(f"duplicate entity mapping for {entity}")
        result[entity] = node
    if not result:
        raise ValueError("entity_to_node must be nonempty")
    return result


def _hex_identifier(value: Any, *, name: str, length: int) -> str:
    if type(value) is not str or len(value) != length:
        raise ValueError(f"{name} must be a {length}-character hexadecimal string")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise ValueError(f"{name} must be a {length}-character hexadecimal string")
    return lowered


@dataclass(frozen=True)
class Prob4DJointObservationDiagnostics:
    """Explicit semantic choices made by the Prob4D adapter."""

    row_count: int
    observation_count: int
    factor_rank: int
    factor_group_count: int
    reliability_policy: Prob4DReliabilityPolicy
    nonneutral_association_count: int
    nonneutral_prior_reliability_count: int
    nonneutral_group_nominal_count: int
    nonunit_group_composite_weight_count: int
    frame_mapping: tuple[tuple[int, int], ...]
    entity_mapping: tuple[tuple[int, int], ...]
    provider_validation: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.row_count < 1 or self.observation_count != 3 * self.row_count:
            raise ValueError("Prob4D diagnostic dimensions are inconsistent")
        if self.factor_rank < 0 or self.factor_group_count != 1:
            raise ValueError("Prob4D diagnostic factor dimensions are invalid")
        if self.reliability_policy not in {"require_neutral", "record_only"}:
            raise ValueError("unsupported Prob4D reliability policy")
        for name in (
            "nonneutral_association_count",
            "nonneutral_prior_reliability_count",
            "nonneutral_group_nominal_count",
            "nonunit_group_composite_weight_count",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not self.frame_mapping or not self.entity_mapping:
            raise ValueError("Prob4D diagnostic mappings must be nonempty")
        validation = validated_json_mapping(
            self.provider_validation,
            error_message="provider validation must contain finite JSON data",
        )
        object.__setattr__(self, "provider_validation", validation)


def joint_observation_from_prob4d(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    rollout_frame_ids: Sequence[int],
    entity_to_node: Mapping[int, int],
    reliability_policy: Prob4DReliabilityPolicy = "require_neutral",
    evidence_id: str | None = None,
) -> tuple[LinearJointObservationEvidence, Prob4DJointObservationDiagnostics]:
    """Convert one strict Prob4D causal observation to scalable joint evidence.

    The existing independent validator is applied before any arrays are adapted.
    Local 3-D covariance blocks remain block diagonal, while the provider's
    shared low-rank factor preserves cross-row covariance. Reliability and
    composite weights are never silently reinterpreted: nonneutral values require
    the explicit exploratory ``record_only`` policy.
    """

    if reliability_policy not in {"require_neutral", "record_only"}:
        raise ValueError("unsupported Prob4D reliability policy")
    validation = validated_json_mapping(
        validate_prob4d_causal_observation_metadata(descriptor, arrays),
        error_message="provider validation must contain finite JSON data",
    )

    means = np.asarray(_required_array(arrays, "mean_xyz_m"), dtype=float)
    if means.ndim != 2 or means.shape[1] != 3 or len(means) == 0:
        raise ValueError("Prob4D mean_xyz_m must have shape (row, 3)")
    if not np.all(np.isfinite(means)):
        raise ValueError("Prob4D mean_xyz_m must be finite")
    row_count = len(means)

    local_covariance = np.asarray(
        _required_array(arrays, "local_covariance_m2"),
        dtype=float,
    )
    if local_covariance.shape != (row_count, 3, 3):
        raise ValueError(
            "Prob4D local_covariance_m2 must have shape (row, 3, 3)"
        )

    low_rank = np.asarray(_required_array(arrays, "low_rank_factor_m"), dtype=float)
    if low_rank.ndim != 3 or low_rank.shape[:2] != (row_count, 3):
        raise ValueError(
            "Prob4D low_rank_factor_m must have shape (row, 3, rank)"
        )
    if not np.all(np.isfinite(low_rank)):
        raise ValueError("Prob4D low_rank_factor_m must be finite")
    factor_rank = low_rank.shape[-1]

    frame_ids = _integer_vector(arrays, "frame_ids", length=row_count)
    entity_ids = _integer_vector(arrays, "entity_ids", length=row_count)
    factor_groups = _integer_vector(arrays, "factor_group_ids", length=row_count)

    association_source = _required_array(arrays, "association_probability")
    prior_reliability_source = _required_array(arrays, "prior_reliability")
    group_nominal_source = _required_array(
        arrays,
        "group_prior_nominal_probability",
    )
    group_weight_source = _required_array(arrays, "group_composite_weight")
    association = _probability_array(
        association_source,
        "association_probability",
        length=row_count,
    )
    prior_reliability = _probability_array(
        prior_reliability_source,
        "prior_reliability",
        length=row_count,
    )
    group_nominal = _probability_array(
        group_nominal_source,
        "group_prior_nominal_probability",
    )
    group_weight = _probability_array(
        group_weight_source,
        "group_composite_weight",
    )
    factor_group_count = len(np.unique(factor_groups))
    if factor_group_count != 1:
        raise ValueError(
            "strict joint Prob4D evidence must use one shared factor group"
        )

    nonneutral_association = int(np.count_nonzero(association != 1.0))
    nonneutral_reliability = int(np.count_nonzero(prior_reliability != 1.0))
    nonneutral_group_nominal = int(np.count_nonzero(group_nominal != 1.0))
    nonunit_group_weight = int(np.count_nonzero(group_weight != 1.0))
    if reliability_policy == "require_neutral" and any(
        (
            nonneutral_association,
            nonneutral_reliability,
            nonneutral_group_nominal,
            nonunit_group_weight,
        )
    ):
        raise ValueError(
            "Prob4D reliability/composite weights are nonneutral; use the "
            "explicit record_only policy only for a registered exploratory path"
        )

    rollout_frames = _normalized_rollout_frames(rollout_frame_ids)
    frame_lookup = {frame_id: index for index, frame_id in enumerate(rollout_frames)}
    missing_frames = sorted(set(map(int, frame_ids)) - set(frame_lookup))
    if missing_frames:
        raise ValueError(
            f"Prob4D rows reference rollout frames not supplied: {missing_frames}"
        )
    mapped_frames = np.asarray(
        [frame_lookup[int(frame)] for frame in frame_ids],
        dtype=np.int64,
    )

    entity_mapping = _normalized_entity_mapping(entity_to_node)
    missing_entities = sorted(set(map(int, entity_ids)) - set(entity_mapping))
    if missing_entities:
        raise ValueError(
            f"Prob4D rows reference unmapped entities: {missing_entities}"
        )
    mapped_nodes = np.asarray(
        [entity_mapping[int(entity)] for entity in entity_ids],
        dtype=np.int64,
    )

    observation_count = 3 * row_count
    shared_factor = None
    if factor_rank:
        shared_factor = low_rank.reshape(observation_count, factor_rank)
    source_artifact_sha256 = _hex_identifier(
        descriptor.get("source_artifact_sha256"),
        name="source_artifact_sha256",
        length=64,
    )
    source_revision = _hex_identifier(
        descriptor.get("source_revision"),
        name="source_revision",
        length=40,
    )
    case_id = descriptor.get("case_id")
    if type(case_id) is not str or not case_id:
        raise ValueError("case_id must be a nonempty string")
    resolved_evidence_id = evidence_id
    if resolved_evidence_id is None:
        resolved_evidence_id = f"prob4d-joint:{case_id}:{source_artifact_sha256}"
    if type(resolved_evidence_id) is not str or not resolved_evidence_id:
        raise ValueError("evidence_id must be a nonempty string")

    used_frame_mapping = tuple(
        sorted((int(frame), int(frame_lookup[int(frame)])) for frame in set(frame_ids))
    )
    used_entity_mapping = tuple(
        sorted(
            (int(entity), int(entity_mapping[int(entity)]))
            for entity in set(entity_ids)
        )
    )
    evidence = LinearJointObservationEvidence(
        evidence_id=resolved_evidence_id,
        values_m=means.reshape(observation_count),
        row_indices=np.arange(observation_count, dtype=np.int64),
        frame_indices=np.repeat(mapped_frames, 3),
        node_indices=np.repeat(mapped_nodes, 3),
        coordinate_indices=np.tile(np.arange(3, dtype=np.int64), row_count),
        coefficients=np.ones(observation_count, dtype=float),
        base_covariance_m2=local_covariance,
        shared_covariance_factor_m=shared_factor,
        source_id="prob4d",
        metadata={
            "adapter_schema_version": PROB4D_JOINT_ADAPTER_SCHEMA_VERSION,
            "adapter": "joint_observation_from_prob4d",
            "case_id": case_id,
            "source_revision": source_revision,
            "source_artifact_sha256": source_artifact_sha256,
            "reliability_policy": reliability_policy,
            "association_probability_sha256": array_sha256(association_source),
            "prior_reliability_sha256": array_sha256(prior_reliability_source),
            "group_prior_nominal_probability_sha256": array_sha256(
                group_nominal_source
            ),
            "group_composite_weight_sha256": array_sha256(group_weight_source),
            "frame_mapping": [list(item) for item in used_frame_mapping],
            "entity_mapping": [list(item) for item in used_entity_mapping],
            "provider_validation": plain_json(validation),
        },
    )
    diagnostics = Prob4DJointObservationDiagnostics(
        row_count=row_count,
        observation_count=observation_count,
        factor_rank=factor_rank,
        factor_group_count=factor_group_count,
        reliability_policy=reliability_policy,
        nonneutral_association_count=nonneutral_association,
        nonneutral_prior_reliability_count=nonneutral_reliability,
        nonneutral_group_nominal_count=nonneutral_group_nominal,
        nonunit_group_composite_weight_count=nonunit_group_weight,
        frame_mapping=used_frame_mapping,
        entity_mapping=used_entity_mapping,
        provider_validation=validation,
    )
    return evidence, diagnostics


__all__ = [
    "PROB4D_JOINT_ADAPTER_SCHEMA_VERSION",
    "Prob4DJointObservationDiagnostics",
    "Prob4DReliabilityPolicy",
    "joint_observation_from_prob4d",
]
