"""Portable, content-addressed external sparse trajectory forecasts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from causal4d.atomic_io import atomic_write_binary
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping

EXTERNAL_FORECAST_SCHEMA = "causal4d.external_sparse_trajectory_forecast"
EXTERNAL_FORECAST_SCHEMA_VERSION = 1
EXTERNAL_FORECAST_IMPORT_SCHEMA = "causal4d.external_forecast_import"
EXTERNAL_FORECAST_IMPORT_SCHEMA_VERSION = 1

_ARCHIVE_FIELDS = frozenset(
    {
        "descriptor_json",
        "node_indices",
        "anchor_positions_m",
        "future_positions_m",
        "physical_frame_indices",
        "validity_mask",
        "future_times_s",
    }
)
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "case_id",
        "source",
        "forecast_ids",
        "anchor_physical_frame",
        "forecast_metadata",
        "metadata",
        "payload",
    }
)
_SOURCE_FIELDS = frozenset({"model", "revision", "artifact_id"})
_PAYLOAD_FIELDS = frozenset(
    {
        "node_indices_sha256",
        "anchor_positions_m_sha256",
        "future_positions_m_sha256",
        "physical_frame_indices_sha256",
        "validity_mask_sha256",
        "future_times_s_sha256",
    }
)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_fields(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, name=name)


def _require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _validated_string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _normalize_forecast_metadata(
    values: Mapping[str, Any],
    forecast_ids: tuple[str, ...],
) -> Mapping[str, Any]:
    supplied = _require_mapping(values, name="forecast_metadata")
    unexpected = sorted(set(supplied) - set(forecast_ids))
    if unexpected:
        raise ValueError(
            "forecast_metadata contains unknown forecast ids: " + repr(unexpected)
        )
    normalized: dict[str, Any] = {}
    for forecast_id in forecast_ids:
        entry = supplied.get(forecast_id, {})
        normalized[forecast_id] = dict(
            _require_mapping(
                entry,
                name=f"forecast_metadata[{forecast_id!r}]",
            )
        )
    return validated_json_mapping(
        normalized,
        error_message="forecast_metadata must be finite JSON data",
    )


@dataclass(frozen=True)
class ExternalForecastBundle:
    """Canonical sparse trajectory forecasts aligned to physical node identities."""

    case_id: str
    source_model: str
    forecast_ids: tuple[str, ...]
    node_indices: np.ndarray
    anchor_positions_m: np.ndarray
    future_positions_m: np.ndarray
    physical_frame_indices: np.ndarray
    validity_mask: np.ndarray | None = None
    anchor_physical_frame: int = 0
    source_revision: str | None = None
    source_artifact_id: str | None = None
    future_times_s: np.ndarray | None = None
    forecast_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = _require_nonempty_string(self.case_id, name="case_id")
        source_model = _require_nonempty_string(
            self.source_model,
            name="source_model",
        )
        forecast_ids = _validated_string_tuple(
            self.forecast_ids,
            name="forecast_ids",
        )
        source_revision = _require_optional_string(
            self.source_revision,
            name="source_revision",
        )
        source_artifact_id = _require_optional_string(
            self.source_artifact_id,
            name="source_artifact_id",
        )
        anchor_frame = _require_integer(
            self.anchor_physical_frame,
            name="anchor_physical_frame",
        )

        nodes = readonly_integer_array(self.node_indices, name="node_indices")
        if nodes.ndim != 1 or not len(nodes):
            raise ValueError("node_indices must be a nonempty vector")
        if np.any(nodes < 0) or len(np.unique(nodes)) != len(nodes):
            raise ValueError("node_indices must be unique and nonnegative")

        anchor = readonly_array(self.anchor_positions_m, dtype=np.float64)
        if anchor.shape != (len(nodes), 3) or not np.all(np.isfinite(anchor)):
            raise ValueError("anchor_positions_m must have finite shape (P, 3)")

        future = np.asarray(self.future_positions_m, dtype=np.float64).copy()
        if future.ndim != 4 or future.shape[3] != 3:
            raise ValueError("future_positions_m must have shape (K, F, P, 3)")
        if future.shape[0] != len(forecast_ids):
            raise ValueError("future_positions_m K must match forecast_ids")
        if future.shape[1] < 1 or future.shape[2] != len(nodes):
            raise ValueError("future_positions_m must have nonempty F and matching P")

        if self.validity_mask is None:
            valid = np.isfinite(future)
        else:
            supplied = np.asarray(self.validity_mask, dtype=bool)
            if supplied.shape == future.shape[:3]:
                supplied = np.repeat(supplied[..., None], 3, axis=3)
            if supplied.shape != future.shape:
                raise ValueError(
                    "validity_mask must have shape (K, F, P) or (K, F, P, 3)"
                )
            valid = supplied.copy()
        if np.any(valid & ~np.isfinite(future)):
            raise ValueError("valid future coordinates must be finite")
        if np.any(np.sum(valid, axis=(1, 2, 3)) == 0):
            raise ValueError("each forecast must contain at least one valid coordinate")
        future[~valid] = np.nan
        future.setflags(write=False)
        valid.setflags(write=False)

        frames = readonly_array(self.physical_frame_indices, dtype=np.float64)
        if frames.shape != (future.shape[1],) or not np.all(np.isfinite(frames)):
            raise ValueError(
                "physical_frame_indices must be a finite vector matching F"
            )
        if np.any(frames <= anchor_frame) or np.any(np.diff(frames) <= 0.0):
            raise ValueError(
                "physical_frame_indices must be strictly increasing after the anchor"
            )

        times = None
        if self.future_times_s is not None:
            times = readonly_array(self.future_times_s, dtype=np.float64)
            if times.shape != (future.shape[1],) or not np.all(np.isfinite(times)):
                raise ValueError("future_times_s must be a finite vector matching F")
            if np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
                raise ValueError(
                    "future_times_s must be strictly increasing and positive"
                )

        forecast_metadata = _normalize_forecast_metadata(
            self.forecast_metadata,
            forecast_ids,
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="metadata must be finite JSON data",
        )

        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "source_model", source_model)
        object.__setattr__(self, "forecast_ids", forecast_ids)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_artifact_id", source_artifact_id)
        object.__setattr__(self, "anchor_physical_frame", anchor_frame)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "anchor_positions_m", anchor)
        object.__setattr__(self, "future_positions_m", future)
        object.__setattr__(self, "physical_frame_indices", frames)
        object.__setattr__(self, "validity_mask", valid)
        object.__setattr__(self, "future_times_s", times)
        object.__setattr__(self, "forecast_metadata", forecast_metadata)
        object.__setattr__(self, "metadata", metadata)

    @property
    def forecast_count(self) -> int:
        return len(self.forecast_ids)

    @property
    def future_horizon(self) -> int:
        return int(self.future_positions_m.shape[1])

    @property
    def point_count(self) -> int:
        return len(self.node_indices)

    @property
    def coordinate_validity(self) -> np.ndarray:
        """Return the normalized coordinate-level validity mask."""

        validity = self.validity_mask
        if validity is None:
            raise RuntimeError("normalized external forecast is missing validity")
        return validity

    def forecast_index(self, forecast_id: str) -> int:
        try:
            return self.forecast_ids.index(forecast_id)
        except ValueError as error:
            raise ValueError(f"unknown external forecast id {forecast_id!r}") from error

    def _payload_hashes(self) -> dict[str, str | None]:
        return {
            "node_indices_sha256": array_sha256(self.node_indices),
            "anchor_positions_m_sha256": array_sha256(self.anchor_positions_m),
            "future_positions_m_sha256": array_sha256(self.future_positions_m),
            "physical_frame_indices_sha256": array_sha256(
                self.physical_frame_indices
            ),
            "validity_mask_sha256": array_sha256(self.coordinate_validity),
            "future_times_s_sha256": (
                array_sha256(self.future_times_s)
                if self.future_times_s is not None
                else None
            ),
        }

    def _descriptor_without_id(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_FORECAST_SCHEMA,
            "schema_version": EXTERNAL_FORECAST_SCHEMA_VERSION,
            "case_id": self.case_id,
            "source": {
                "model": self.source_model,
                "revision": self.source_revision,
                "artifact_id": self.source_artifact_id,
            },
            "forecast_ids": list(self.forecast_ids),
            "anchor_physical_frame": self.anchor_physical_frame,
            "forecast_metadata": plain_json(self.forecast_metadata),
            "metadata": plain_json(self.metadata),
            "payload": self._payload_hashes(),
        }

    @property
    def artifact_id(self) -> str:
        encoded = _canonical_json(self._descriptor_without_id()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        descriptor = self._descriptor_without_id()
        descriptor["artifact_id"] = self.artifact_id
        return descriptor

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "case_id": self.case_id,
            "source_model": self.source_model,
            "source_revision": self.source_revision,
            "forecast_ids": list(self.forecast_ids),
            "forecast_count": self.forecast_count,
            "future_horizon": self.future_horizon,
            "point_count": self.point_count,
            "anchor_physical_frame": self.anchor_physical_frame,
            "physical_frame_start": float(self.physical_frame_indices[0]),
            "physical_frame_stop": float(self.physical_frame_indices[-1]),
            "valid_coordinate_fraction": float(np.mean(self.coordinate_validity)),
        }


def _save_archive(handle: BinaryIO, bundle: ExternalForecastBundle) -> None:
    np.savez_compressed(
        handle,
        descriptor_json=np.asarray(_canonical_json(bundle.descriptor())),
        node_indices=bundle.node_indices,
        anchor_positions_m=bundle.anchor_positions_m,
        future_positions_m=bundle.future_positions_m,
        physical_frame_indices=bundle.physical_frame_indices,
        validity_mask=bundle.coordinate_validity,
        future_times_s=(
            bundle.future_times_s
            if bundle.future_times_s is not None
            else np.asarray([], dtype=np.float64)
        ),
    )


def save_external_forecast(
    path: str | Path,
    bundle: ExternalForecastBundle,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically publish a canonical external-forecast artifact."""

    expected_id = bundle.artifact_id

    def validate(temporary: Path) -> None:
        loaded = load_external_forecast(temporary)
        if loaded.artifact_id != expected_id:
            raise ValueError("external forecast changed during serialization")

    atomic_write_binary(
        path,
        lambda handle: _save_archive(handle, bundle),
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
    try:
        parsed = json.loads(
            scalar,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("descriptor_json is invalid") from error
    return _require_exact_fields(
        parsed,
        name="external forecast descriptor",
        required=_DESCRIPTOR_FIELDS,
    )


def is_external_forecast_artifact(path: str | Path) -> bool:
    """Return whether an NPZ advertises the canonical external-forecast schema."""

    with np.load(path, allow_pickle=False) as archive:
        if "descriptor_json" not in archive.files:
            return False
        descriptor = _parse_descriptor(archive["descriptor_json"])
        return descriptor["schema"] == EXTERNAL_FORECAST_SCHEMA


def load_external_forecast(path: str | Path) -> ExternalForecastBundle:
    """Load and fully revalidate a canonical external-forecast artifact."""

    with np.load(path, allow_pickle=False) as archive:
        actual_fields = frozenset(archive.files)
        if actual_fields != _ARCHIVE_FIELDS:
            raise ValueError(
                "external forecast archive fields do not match schema; "
                f"missing={sorted(_ARCHIVE_FIELDS - actual_fields)}, "
                f"unexpected={sorted(actual_fields - _ARCHIVE_FIELDS)}"
            )
        descriptor = _parse_descriptor(archive["descriptor_json"])
        if _require_nonempty_string(
            descriptor["schema"],
            name="external forecast schema",
        ) != EXTERNAL_FORECAST_SCHEMA:
            raise ValueError("artifact is not a Causal4D external forecast")
        if _require_integer(
            descriptor["schema_version"],
            name="external forecast schema_version",
            minimum=1,
        ) != EXTERNAL_FORECAST_SCHEMA_VERSION:
            raise ValueError("unsupported external forecast schema version")
        source = _require_exact_fields(
            descriptor["source"],
            name="external forecast source",
            required=_SOURCE_FIELDS,
        )
        payload = _require_exact_fields(
            descriptor["payload"],
            name="external forecast payload",
            required=_PAYLOAD_FIELDS,
        )
        times_array = np.asarray(archive["future_times_s"], dtype=np.float64)
        if times_array.ndim != 1:
            raise ValueError("future_times_s archive array must be one-dimensional")
        bundle = ExternalForecastBundle(
            case_id=descriptor["case_id"],
            source_model=source["model"],
            source_revision=source["revision"],
            source_artifact_id=source["artifact_id"],
            forecast_ids=tuple(descriptor["forecast_ids"]),
            node_indices=np.asarray(archive["node_indices"]),
            anchor_positions_m=np.asarray(archive["anchor_positions_m"]),
            future_positions_m=np.asarray(archive["future_positions_m"]),
            physical_frame_indices=np.asarray(archive["physical_frame_indices"]),
            validity_mask=np.asarray(archive["validity_mask"]),
            anchor_physical_frame=descriptor["anchor_physical_frame"],
            future_times_s=times_array if len(times_array) else None,
            forecast_metadata=descriptor["forecast_metadata"],
            metadata=descriptor["metadata"],
        )
        expected_artifact_id = _validate_sha256(
            descriptor["artifact_id"],
            name="external forecast artifact_id",
        )
        if bundle.artifact_id != expected_artifact_id:
            raise ValueError("external forecast artifact_id does not match payload")
        if bundle._payload_hashes() != dict(payload):
            raise ValueError("external forecast payload hashes do not match arrays")
        return bundle


def import_external_forecast(
    source_npz: str | Path,
    import_manifest_json: str | Path,
) -> ExternalForecastBundle:
    """Import a producer artifact through the strict manifest adapter."""

    from causal4d.external_forecast_importer import import_external_forecast as impl

    return impl(source_npz, import_manifest_json)


__all__ = [
    "EXTERNAL_FORECAST_IMPORT_SCHEMA",
    "EXTERNAL_FORECAST_IMPORT_SCHEMA_VERSION",
    "EXTERNAL_FORECAST_SCHEMA",
    "EXTERNAL_FORECAST_SCHEMA_VERSION",
    "ExternalForecastBundle",
    "import_external_forecast",
    "is_external_forecast_artifact",
    "load_external_forecast",
    "save_external_forecast",
]
