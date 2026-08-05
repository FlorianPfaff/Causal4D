"""Shared validation and content-identity helpers for multi-contact paths."""

from __future__ import annotations

import hashlib
import json
from numbers import Real
from typing import Any, Sequence

import numpy as np

from causal4d.immutable_array import readonly_array, readonly_integer_array


MULTI_CONTACT_SCHEDULE_SCHEMA_VERSION = "causal4d.multi_contact_schedule.v1"


def readonly(values: Any, *, dtype: Any = None) -> np.ndarray:
    """Return a defensive, immutable NumPy copy."""

    return readonly_array(values, dtype=dtype)


def real_array(
    values: Any,
    *,
    name: str,
    require_finite: bool = True,
) -> np.ndarray:
    """Return owned real-valued data without Boolean or string coercion."""

    raw = np.asarray(values)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numbers")
    result = raw.astype(float, copy=True)
    if require_finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def integer_array(
    values: Any,
    *,
    name: str,
    dtype: Any = np.int64,
) -> np.ndarray:
    """Return immutable integer data without truncating or coercing inputs."""

    exact = readonly_integer_array(values, name=name)
    target_dtype = np.dtype(dtype)
    if target_dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} target dtype must be integral")
    if exact.size:
        limits = np.iinfo(target_dtype)
        if int(np.min(exact)) < limits.min or int(np.max(exact)) > limits.max:
            raise ValueError(
                f"{name} contains an integer outside {target_dtype.name} range"
            )
    return readonly(exact, dtype=target_dtype)


def normalized_weights(values: Any, *, name: str) -> np.ndarray:
    """Validate and normalize a finite nonnegative weight vector."""

    weights = real_array(values, name=name)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive mass")
    tolerance = 16.0 * np.finfo(float).eps * max(1, len(weights))
    normalized = weights if abs(total - 1.0) <= tolerance else weights / total
    return readonly(normalized)


def probability_mass(value: object, *, name: str) -> float:
    """Validate one finite probability mass, allowing only round-off above one."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real probability mass")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result <= 1.0 + 1e-12:
        raise ValueError(f"{name} must lie in (0, 1]")
    return min(result, 1.0)


def activation_matrix(
    values: Any,
    *,
    contact_count: int | None = None,
    frame_count: int | None = None,
    name: str = "command_activation",
) -> np.ndarray:
    """Validate a single- or multi-contact activation schedule."""

    activation = real_array(values, name=name)
    if activation.ndim == 1:
        activation = activation[None, :]
    if activation.ndim != 2 or min(activation.shape) < 1:
        raise ValueError(f"{name} must have shape (G, T) or (T,)")
    if contact_count is not None and activation.shape[0] != contact_count:
        raise ValueError(f"{name} must identify every contact channel")
    if frame_count is not None and activation.shape[1] != frame_count:
        raise ValueError(f"{name} must match the rollout frame count")
    if np.any((activation < 0.0) | (activation > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return activation


def validated_contact_ids(
    values: Sequence[str] | None,
    contact_count: int,
) -> tuple[str, ...]:
    """Return nonempty unique contact identifiers without string coercion."""

    if contact_count < 1:
        raise ValueError("contact paths must identify at least one contact channel")
    if values is None:
        return tuple(f"contact-{index}" for index in range(contact_count))
    return validate_identifiers(
        values,
        expected_count=contact_count,
        name="contact_ids",
    )


def validate_identifiers(
    values: Sequence[str],
    *,
    expected_count: int,
    name: str,
) -> tuple[str, ...]:
    """Validate and freeze identifiers against their support size."""

    if expected_count < 1 or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be nonempty unique strings")
    result = tuple(values)
    if len(result) != expected_count or any(
        type(value) is not str or not value for value in result
    ) or len(set(result)) != expected_count:
        raise ValueError(f"{name} must be nonempty unique strings")
    return result


def schedule_identity(
    contact_ids: tuple[str, ...],
    path_ids: tuple[str, ...],
    regime_paths: np.ndarray,
    weights: np.ndarray,
    retained_prior_mass: float,
) -> str:
    """Return a stable SHA-256 identity for one retained schedule support."""

    payload = {
        "schema_version": MULTI_CONTACT_SCHEDULE_SCHEMA_VERSION,
        "contact_ids": list(contact_ids),
        "path_ids": list(path_ids),
        "regime_paths": np.asarray(regime_paths, dtype=np.int8).tolist(),
        "weights_hex": [
            float(value).hex() for value in np.asarray(weights, dtype=float)
        ],
        "retained_prior_mass_hex": float(retained_prior_mass).hex(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
