"""Shared validation helpers for projected functional-support contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.immutable_json import plain_json, validated_json_mapping


PROJECTED_FUNCTIONAL_SUPPORT_SCHEMA_VERSION = 1
_FORBIDDEN_SOURCE_METADATA_KEYS = {
    "evaluation_target",
    "held_out_target",
    "target_continuation",
    "target_future",
    "target_future_observations",
    "target_future_outcomes",
    "target_loss",
    "target_outcome",
    "target_outcomes",
    "target_outcomes_used",
    "target_value",
    "target_values",
}


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def require_sha256(value: Any, *, name: str) -> str:
    digest = require_nonempty_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def validated_string_tuple(
    values: Any,
    *,
    name: str,
    require_digest: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    validator = require_sha256 if require_digest else require_nonempty_string
    result = tuple(
        validator(value, name=f"{name}[{index}]") for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def finite_nonnegative_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def finite_positive_float(value: Any, *, name: str) -> float:
    result = finite_nonnegative_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def require_no_target_access(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    if value:
        raise ValueError(f"{name} must be false for source-only certification")
    return False


def _reject_target_outcome_metadata(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_SOURCE_METADATA_KEYS:
                raise ValueError(
                    f"{path}.{key} is forbidden for source-only certification"
                )
            _reject_target_outcome_metadata(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_target_outcome_metadata(item, path=f"{path}[{index}]")


def validated_source_metadata(
    values: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    metadata = validated_json_mapping(
        values,
        error_message=f"{name} must contain finite JSON data",
    )
    _reject_target_outcome_metadata(metadata, path=name)
    return metadata
