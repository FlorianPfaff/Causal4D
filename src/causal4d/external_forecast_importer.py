"""Strict adapter from producer NPZ files to external forecast artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.external_forecast import (
    EXTERNAL_FORECAST_IMPORT_SCHEMA,
    EXTERNAL_FORECAST_IMPORT_SCHEMA_VERSION,
    ExternalForecastBundle,
)
from causal4d.immutable_json import plain_json, validated_json_mapping

_IMPORT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "case_id",
        "source",
        "arrays",
        "layout",
        "coordinate_frame",
        "position_unit",
        "forecast_ids",
    }
)
_IMPORT_OPTIONAL_FIELDS = frozenset(
    {
        "anchor_physical_frame",
        "physical_fps",
        "forecast_metadata",
        "metadata",
    }
)
_IMPORT_ARRAY_FIELDS = frozenset(
    {"node_indices", "anchor_positions", "future_positions"}
)
_IMPORT_ARRAY_OPTIONAL_FIELDS = frozenset(
    {
        "physical_frame_indices",
        "future_times_s",
        "validity_mask",
        "camera_to_world",
    }
)
_UNIT_SCALE_TO_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
_LAYOUTS = frozenset({"PFC", "FPC", "KPFC", "KFPC"})
_COORDINATE_FRAMES = frozenset({"world", "camera"})


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
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required - optional)
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


def _require_positive_number(value: Any, *, name: str) -> float:
    if type(value) not in {int, float} or not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


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


def _load_json_mapping(path: str | Path, *, name: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    return _require_mapping(parsed, name=name)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_array(
    archive: np.lib.npyio.NpzFile,
    key: Any,
    *,
    name: str,
) -> np.ndarray:
    key_text = _require_nonempty_string(key, name=f"arrays.{name}")
    if key_text not in archive.files:
        raise ValueError(f"source NPZ does not contain array {key_text!r} for {name}")
    try:
        return np.asarray(archive[key_text])
    except ValueError as error:
        raise ValueError(
            f"source array {key_text!r} cannot be loaded without pickle"
        ) from error


def _normalize_future_positions(values: np.ndarray, layout: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if layout == "PFC":
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("PFC future_positions must have shape (P, F, 3)")
        return np.transpose(array, (1, 0, 2))[None]
    if layout == "FPC":
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("FPC future_positions must have shape (F, P, 3)")
        return array[None]
    if layout == "KPFC":
        if array.ndim != 4 or array.shape[3] != 3:
            raise ValueError("KPFC future_positions must have shape (K, P, F, 3)")
        return np.transpose(array, (0, 2, 1, 3))
    if layout == "KFPC":
        if array.ndim != 4 or array.shape[3] != 3:
            raise ValueError("KFPC future_positions must have shape (K, F, P, 3)")
        return array
    raise ValueError(f"unsupported future_positions layout {layout!r}")


def _normalize_validity(
    values: np.ndarray,
    *,
    layout: str,
    canonical_shape: tuple[int, int, int, int],
) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    forecast_count, frame_count, point_count, _ = canonical_shape
    if layout == "PFC":
        point_shape = (point_count, frame_count)
        coordinate_shape = (*point_shape, 3)
        if array.shape == point_shape:
            result = np.transpose(array, (1, 0))[None, ..., None]
        elif array.shape == coordinate_shape:
            result = np.transpose(array, (1, 0, 2))[None]
        else:
            raise ValueError(
                "PFC validity must have shape (P, F) or (P, F, 3)"
            )
    elif layout == "FPC":
        point_shape = (frame_count, point_count)
        coordinate_shape = (*point_shape, 3)
        if array.shape == point_shape:
            result = array[None, ..., None]
        elif array.shape == coordinate_shape:
            result = array[None]
        else:
            raise ValueError(
                "FPC validity must have shape (F, P) or (F, P, 3)"
            )
    elif layout == "KPFC":
        point_shape = (forecast_count, point_count, frame_count)
        coordinate_shape = (*point_shape, 3)
        if array.shape == point_shape:
            result = np.transpose(array, (0, 2, 1))[..., None]
        elif array.shape == coordinate_shape:
            result = np.transpose(array, (0, 2, 1, 3))
        else:
            raise ValueError(
                "KPFC validity must have shape (K, P, F) or (K, P, F, 3)"
            )
    elif layout == "KFPC":
        point_shape = (forecast_count, frame_count, point_count)
        coordinate_shape = (*point_shape, 3)
        if array.shape == point_shape:
            result = array[..., None]
        elif array.shape == coordinate_shape:
            result = array
        else:
            raise ValueError(
                "KFPC validity must have shape (K, F, P) or (K, F, P, 3)"
            )
    else:
        raise ValueError(f"unsupported validity layout {layout!r}")
    if result.shape[-1] == 1:
        result = np.repeat(result, 3, axis=3)
    if result.shape != canonical_shape:
        raise ValueError(
            "validity_mask shape does not match normalized future_positions: "
            f"{result.shape} != {canonical_shape}"
        )
    return result


def _camera_to_world(points_m: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_to_world must have finite shape (4, 4)")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise ValueError("camera_to_world must be a homogeneous rigid transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=1e-6):
        raise ValueError("camera_to_world rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("camera_to_world rotation must have determinant one")
    flat = points_m.reshape(-1, 3)
    homogeneous = np.column_stack((flat, np.ones(len(flat))))
    transformed = homogeneous @ matrix.T
    return transformed[:, :3].reshape(points_m.shape)


def import_external_forecast(
    source_npz: str | Path,
    import_manifest_json: str | Path,
) -> ExternalForecastBundle:
    """Normalize a producer NPZ according to a strict, portable manifest."""

    manifest_hash = _file_sha256(import_manifest_json)
    manifest = _require_exact_fields(
        _load_json_mapping(import_manifest_json, name="external forecast manifest"),
        name="external forecast manifest",
        required=_IMPORT_FIELDS,
        optional=_IMPORT_OPTIONAL_FIELDS,
    )
    if _file_sha256(import_manifest_json) != manifest_hash:
        raise ValueError("external forecast manifest changed during import")
    if _require_nonempty_string(
        manifest["schema"],
        name="external forecast import schema",
    ) != EXTERNAL_FORECAST_IMPORT_SCHEMA:
        raise ValueError("unexpected external forecast import schema")
    if _require_integer(
        manifest["schema_version"],
        name="external forecast import schema_version",
        minimum=1,
    ) != EXTERNAL_FORECAST_IMPORT_SCHEMA_VERSION:
        raise ValueError("unsupported external forecast import schema version")

    case_id = _require_nonempty_string(manifest["case_id"], name="case_id")
    source = _require_exact_fields(
        manifest["source"],
        name="external forecast source",
        required=frozenset({"model"}),
        optional=frozenset({"revision", "artifact_id"}),
    )
    source_model = _require_nonempty_string(source["model"], name="source.model")
    source_revision = _require_optional_string(
        source.get("revision"),
        name="source.revision",
    )
    source_artifact_id = _require_optional_string(
        source.get("artifact_id"),
        name="source.artifact_id",
    )
    arrays = _require_exact_fields(
        manifest["arrays"],
        name="external forecast arrays",
        required=_IMPORT_ARRAY_FIELDS,
        optional=_IMPORT_ARRAY_OPTIONAL_FIELDS,
    )
    layout = _require_nonempty_string(manifest["layout"], name="layout")
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {sorted(_LAYOUTS)}")
    coordinate_frame = _require_nonempty_string(
        manifest["coordinate_frame"],
        name="coordinate_frame",
    )
    if coordinate_frame not in _COORDINATE_FRAMES:
        raise ValueError(
            f"coordinate_frame must be one of {sorted(_COORDINATE_FRAMES)}"
        )
    position_unit = _require_nonempty_string(
        manifest["position_unit"],
        name="position_unit",
    )
    if position_unit not in _UNIT_SCALE_TO_M:
        raise ValueError(
            f"position_unit must be one of {sorted(_UNIT_SCALE_TO_M)}"
        )
    forecast_ids = _validated_string_tuple(
        manifest["forecast_ids"],
        name="forecast_ids",
    )
    anchor_frame = _require_integer(
        manifest.get("anchor_physical_frame", 0),
        name="anchor_physical_frame",
    )
    physical_fps = (
        _require_positive_number(manifest["physical_fps"], name="physical_fps")
        if "physical_fps" in manifest
        else None
    )
    forecast_metadata = _require_mapping(
        manifest.get("forecast_metadata", {}),
        name="forecast_metadata",
    )
    producer_metadata = validated_json_mapping(
        _require_mapping(manifest.get("metadata", {}), name="metadata"),
        error_message="manifest metadata must be finite JSON data",
    )

    source_npz_hash = _file_sha256(source_npz)
    with np.load(source_npz, allow_pickle=False) as archive:
        nodes = _source_array(
            archive,
            arrays["node_indices"],
            name="node_indices",
        )
        anchor = np.asarray(
            _source_array(
                archive,
                arrays["anchor_positions"],
                name="anchor_positions",
            ),
            dtype=np.float64,
        )
        future = _normalize_future_positions(
            _source_array(
                archive,
                arrays["future_positions"],
                name="future_positions",
            ),
            layout,
        )
        valid = None
        if "validity_mask" in arrays:
            valid = _normalize_validity(
                _source_array(
                    archive,
                    arrays["validity_mask"],
                    name="validity_mask",
                ),
                layout=layout,
                canonical_shape=future.shape,
            )

        times = None
        if "future_times_s" in arrays:
            times = np.asarray(
                _source_array(
                    archive,
                    arrays["future_times_s"],
                    name="future_times_s",
                ),
                dtype=np.float64,
            )
        frames = None
        if "physical_frame_indices" in arrays:
            frames = np.asarray(
                _source_array(
                    archive,
                    arrays["physical_frame_indices"],
                    name="physical_frame_indices",
                ),
                dtype=np.float64,
            )
        if times is not None and physical_fps is None:
            raise ValueError("physical_fps is required with arrays.future_times_s")
        if frames is None:
            if times is None:
                raise ValueError(
                    "provide arrays.physical_frame_indices or both "
                    "arrays.future_times_s and physical_fps"
                )
            frames = anchor_frame + times * physical_fps
        elif times is not None:
            derived = anchor_frame + times * physical_fps
            if frames.shape != derived.shape or not np.allclose(
                frames,
                derived,
                atol=1e-9,
                rtol=1e-9,
            ):
                raise ValueError(
                    "physical_frame_indices disagree with future_times_s * physical_fps"
                )

        scale = _UNIT_SCALE_TO_M[position_unit]
        anchor = anchor * scale
        future = future * scale
        camera_to_world_hash = None
        if coordinate_frame == "camera":
            if "camera_to_world" not in arrays:
                raise ValueError(
                    "camera-frame imports require arrays.camera_to_world"
                )
            transform = np.asarray(
                _source_array(
                    archive,
                    arrays["camera_to_world"],
                    name="camera_to_world",
                ),
                dtype=np.float64,
            )
            camera_to_world_hash = array_sha256(transform)
            anchor = _camera_to_world(anchor, transform)
            future = _camera_to_world(future, transform)
        elif "camera_to_world" in arrays:
            raise ValueError(
                "arrays.camera_to_world is only valid for coordinate_frame='camera'"
            )

    if _file_sha256(source_npz) != source_npz_hash:
        raise ValueError("source NPZ changed during import")
    import_metadata = {
        "importer": {
            "schema": EXTERNAL_FORECAST_IMPORT_SCHEMA,
            "schema_version": EXTERNAL_FORECAST_IMPORT_SCHEMA_VERSION,
            "source_npz_sha256": source_npz_hash,
            "import_manifest_sha256": manifest_hash,
            "source_layout": layout,
            "source_coordinate_frame": coordinate_frame,
            "source_position_unit": position_unit,
            "camera_to_world_sha256": camera_to_world_hash,
            "physical_fps": physical_fps,
        },
        "producer": plain_json(producer_metadata),
    }
    return ExternalForecastBundle(
        case_id=case_id,
        source_model=source_model,
        source_revision=source_revision,
        source_artifact_id=source_artifact_id,
        forecast_ids=forecast_ids,
        node_indices=nodes,
        anchor_positions_m=anchor,
        future_positions_m=future,
        physical_frame_indices=frames,
        validity_mask=valid,
        anchor_physical_frame=anchor_frame,
        future_times_s=times,
        forecast_metadata=forecast_metadata,
        metadata=import_metadata,
    )


__all__ = ["import_external_forecast"]
