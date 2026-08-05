"""Shared strict-contract helpers for held-out physical target evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from causal4d.contracts import CausalContext
from causal4d.immutable_json import plain_json, validated_json_mapping

HELD_OUT_PHYSICAL_TARGET_SCHEMA = "causal4d.held_out_physical_target"
HELD_OUT_PHYSICAL_TARGET_SCHEMA_VERSION = 1
NODE_ORDER_DENSE_PREFIX_V1 = "dense_zero_based_prefix_v1"

DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "context",
        "source_query_id",
        "trajectory_frame_interval",
        "node_order",
        "source",
        "metadata",
        "payload",
    }
)
SOURCE_FIELDS = frozenset({"kind", "revision", "content_sha256", "artifact_id"})
PAYLOAD_FIELDS = frozenset(
    {"node_indices_sha256", "positions_m_sha256", "validity_mask_sha256"}
)
LOWER_HEX = frozenset("0123456789abcdef")


def require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def require_exact_fields(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    mapping = require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mapping


def require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def require_optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return require_nonempty_string(value, name=name)


def require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def validate_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def require_integer_interval(value: Any, *, name: str) -> list[int]:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(item) is not int for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise ValueError(f"{name} must contain a nonempty integer interval")
    return value


def require_finite_number(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def target_validity(visible: np.ndarray, motion_valid: np.ndarray) -> np.ndarray:
    """Reproduce the public BayesianPhysTwin target-validity convention."""

    visible_array = np.asarray(visible)
    motion_array = np.asarray(motion_valid)
    if visible_array.dtype.kind != "b" or motion_array.dtype.kind != "b":
        raise ValueError("legacy visibility and motion-valid arrays must be boolean")
    if visible_array.ndim != 2 or visible_array.shape[0] < 1:
        raise ValueError("legacy visibility must have shape (T>=1, N)")
    frame_count, track_count = visible_array.shape
    if motion_array.shape not in {
        (frame_count, track_count),
        (frame_count - 1, track_count),
    }:
        raise ValueError("legacy motion-valid array has an incompatible shape")
    valid = np.zeros_like(visible_array, dtype=bool)
    valid[0] = visible_array[0]
    valid[1:] = motion_array[: frame_count - 1]
    return valid


def validate_target_descriptor(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the self-contained identity of a target descriptor."""

    fields = require_exact_fields(
        values,
        name="held-out target descriptor",
        required=DESCRIPTOR_FIELDS,
    )
    if fields["schema"] != HELD_OUT_PHYSICAL_TARGET_SCHEMA:
        raise ValueError("unsupported held-out target schema")
    schema_version = require_integer(
        fields["schema_version"],
        name="held-out target schema_version",
        minimum=1,
    )
    if schema_version != HELD_OUT_PHYSICAL_TARGET_SCHEMA_VERSION:
        raise ValueError("unsupported held-out target schema version")
    if fields["node_order"] != NODE_ORDER_DENSE_PREFIX_V1:
        raise ValueError("unsupported held-out target node order")

    context = CausalContext.from_dict(fields["context"])
    source_query_id = validate_sha256(
        fields["source_query_id"],
        name="source_query_id",
    )
    interval = require_integer_interval(
        fields["trajectory_frame_interval"],
        name="trajectory_frame_interval",
    )
    expected_interval = [
        context.o_minus.frame_stop - 1,
        context.u_cf.frame_stop,
    ]
    if interval != expected_interval:
        raise ValueError("held-out target interval disagrees with its causal context")

    source = require_exact_fields(
        fields["source"],
        name="held-out target source",
        required=SOURCE_FIELDS,
    )
    require_nonempty_string(source["kind"], name="source.kind")
    require_nonempty_string(source["revision"], name="source.revision")
    validate_sha256(source["content_sha256"], name="source.content_sha256")
    require_optional_string(source["artifact_id"], name="source.artifact_id")

    payload = require_exact_fields(
        fields["payload"],
        name="held-out target payload",
        required=PAYLOAD_FIELDS,
    )
    for name, value in payload.items():
        validate_sha256(value, name=f"payload.{name}")
    require_mapping(fields["metadata"], name="held-out target metadata")

    normalized = plain_json(
        validated_json_mapping(
            fields,
            error_message="held-out target descriptor must be finite JSON data",
        )
    )
    supplied_id = normalized.pop("artifact_id")
    validate_sha256(supplied_id, name="artifact_id")
    expected_id = hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()
    if supplied_id != expected_id:
        raise ValueError("held-out target artifact_id does not match descriptor")
    normalized["artifact_id"] = supplied_id
    if normalized["source_query_id"] != source_query_id:
        raise ValueError("held-out target source_query_id changed during validation")
    return normalized
