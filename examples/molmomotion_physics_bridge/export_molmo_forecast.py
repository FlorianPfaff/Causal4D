"""Package MolmoMotion trajectories for Causal4D's external-forecast importer.

The helper is intentionally standalone: it depends only on NumPy and can run in
an existing MolmoMotion environment without installing Causal4D. It accepts a
small NPZ with metric world-frame query trajectories and emits the producer NPZ
plus strict JSON manifest consumed by ``causal4d.external_forecast``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _forecast(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("forecasts must use ID=CAPTION")
    identifier, caption = value.split("=", 1)
    identifier = identifier.strip()
    caption = caption.strip()
    if not identifier or not caption:
        raise argparse.ArgumentTypeError("forecast id and caption must be nonempty")
    return identifier, caption


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _load_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    *,
    name: str,
) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"input NPZ is missing {name} array {key!r}")
    try:
        return np.asarray(archive[key])
    except ValueError as error:
        raise ValueError(f"{name} cannot be loaded without pickle") from error


def _normalize_validity(
    supplied: np.ndarray | None,
    future: np.ndarray,
) -> np.ndarray:
    if supplied is None:
        valid = np.isfinite(future)
    else:
        valid = np.asarray(supplied, dtype=bool)
        if valid.shape == future.shape[:3]:
            valid = np.repeat(valid[..., None], 3, axis=3)
        if valid.shape != future.shape:
            raise ValueError("validity_mask must have shape (K, P, F) or (K, P, F, 3)")
        valid = valid.copy()
    if np.any(valid & ~np.isfinite(future)):
        raise ValueError("coordinates marked valid must be finite")
    if np.any(np.sum(valid, axis=(1, 2, 3)) == 0):
        raise ValueError("every forecast must contain at least one valid coordinate")
    return valid


def _atomic_write_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(
    path: Path, value: Mapping[str, Any], *, overwrite: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_bridge_package(
    input_npz: str | Path,
    output_npz: str | Path,
    output_manifest_json: str | Path,
    *,
    case_id: str,
    source_model: str,
    source_revision: str,
    source_artifact_id: str | None,
    forecasts: Sequence[tuple[str, str]],
    anchor_physical_frame: int,
    physical_fps: float,
    forecast_fps: float,
    producer_environment: str,
    node_key: str = "node_indices",
    anchor_key: str = "anchor_positions_world_m",
    future_key: str = "future_positions_world_m",
    validity_key: str = "validity_mask",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a portable producer NPZ and its strict import manifest.

    The input future array must use ``(forecast, point, future, xyz)`` order.
    Output coordinates remain in metres in the physical world frame.
    """

    for name, value in (
        ("case_id", case_id),
        ("source_model", source_model),
        ("source_revision", source_revision),
        ("producer_environment", producer_environment),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if source_artifact_id is not None and (
        type(source_artifact_id) is not str or not source_artifact_id.strip()
    ):
        raise ValueError("source_artifact_id must be None or a nonempty string")
    if type(anchor_physical_frame) is not int or anchor_physical_frame < 0:
        raise ValueError("anchor_physical_frame must be a nonnegative integer")
    if not np.isfinite(physical_fps) or physical_fps <= 0.0:
        raise ValueError("physical_fps must be finite and positive")
    if not np.isfinite(forecast_fps) or forecast_fps <= 0.0:
        raise ValueError("forecast_fps must be finite and positive")

    forecast_entries = tuple(forecasts)
    if not forecast_entries:
        raise ValueError("at least one forecast is required")
    forecast_ids = tuple(identifier for identifier, _ in forecast_entries)
    if any(not identifier or not caption for identifier, caption in forecast_entries):
        raise ValueError("forecast identifiers and captions must be nonempty")
    if len(set(forecast_ids)) != len(forecast_ids):
        raise ValueError("forecast identifiers must be unique")

    input_path = Path(input_npz).resolve()
    output_path = Path(output_npz).resolve()
    manifest_path = Path(output_manifest_json).resolve()
    if input_path == output_path:
        raise ValueError("input_npz and output_npz must be different files")
    if manifest_path in {input_path, output_path}:
        raise ValueError("output manifest must be a separate file")

    with np.load(input_path, allow_pickle=False) as archive:
        node_values = _load_array(archive, node_key, name="node_indices")
        if not np.issubdtype(node_values.dtype, np.integer):
            raise ValueError("node_indices must use an integer dtype")
        nodes = np.asarray(node_values, dtype=np.int64)
        anchor = np.asarray(
            _load_array(archive, anchor_key, name="anchor_positions"),
            dtype=np.float64,
        )
        future = np.asarray(
            _load_array(archive, future_key, name="future_positions"),
            dtype=np.float64,
        )
        supplied_validity = (
            _load_array(archive, validity_key, name="validity_mask")
            if validity_key in archive.files
            else None
        )

    if nodes.ndim != 1 or not len(nodes):
        raise ValueError("node_indices must be a nonempty vector")
    if np.any(nodes < 0) or len(np.unique(nodes)) != len(nodes):
        raise ValueError("node_indices must be unique and nonnegative")
    if anchor.shape != (len(nodes), 3) or not np.all(np.isfinite(anchor)):
        raise ValueError("anchor_positions_world_m must have finite shape (P, 3)")
    if future.ndim != 4 or future.shape[3] != 3:
        raise ValueError("future_positions_world_m must have shape (K, P, F, 3)")
    if future.shape[0] != len(forecast_entries):
        raise ValueError("future forecast count must match repeated --forecast values")
    if future.shape[1] != len(nodes) or future.shape[2] < 1:
        raise ValueError("future positions must have matching P and nonempty F")

    validity = _normalize_validity(supplied_validity, future)
    future = future.copy()
    future[~validity] = np.nan
    future_count = int(future.shape[2])
    future_times_s = np.arange(1, future_count + 1, dtype=np.float64) / float(
        forecast_fps
    )
    physical_frames = float(anchor_physical_frame) + future_times_s * float(
        physical_fps
    )

    producer_arrays = {
        "node_indices": nodes,
        "anchor_positions_world_m": anchor,
        "future_positions_world_m": future,
        "future_times_s": future_times_s,
        "validity_mask": validity,
    }
    _atomic_write_npz(output_path, producer_arrays, overwrite=overwrite)

    source: dict[str, Any] = {
        "model": source_model.strip(),
        "revision": source_revision.strip(),
    }
    if source_artifact_id is not None:
        source["artifact_id"] = source_artifact_id.strip()
    manifest: dict[str, Any] = {
        "schema": "causal4d.external_forecast_import",
        "schema_version": 1,
        "case_id": case_id.strip(),
        "source": source,
        "arrays": {
            "node_indices": "node_indices",
            "anchor_positions": "anchor_positions_world_m",
            "future_positions": "future_positions_world_m",
            "future_times_s": "future_times_s",
            "validity_mask": "validity_mask",
        },
        "layout": "KPFC",
        "coordinate_frame": "world",
        "position_unit": "m",
        "forecast_ids": list(forecast_ids),
        "anchor_physical_frame": anchor_physical_frame,
        "physical_fps": float(physical_fps),
        "forecast_metadata": {
            identifier: {"caption": caption} for identifier, caption in forecast_entries
        },
        "metadata": {
            "bridge_helper": "molmomotion_physics_bridge_v1",
            "forecast_fps": float(forecast_fps),
            "producer_environment": producer_environment.strip(),
        },
    }
    try:
        _atomic_write_json(manifest_path, manifest, overwrite=overwrite)
    except Exception:
        if not overwrite:
            output_path.unlink(missing_ok=True)
        raise

    return {
        "anchor_physical_frame": anchor_physical_frame,
        "forecast_ids": list(forecast_ids),
        "forecast_shape_kpfc": list(future.shape),
        "future_times_s": future_times_s.tolist(),
        "manifest": str(manifest_path),
        "node_count": int(len(nodes)),
        "physical_frame_indices": physical_frames.tolist(),
        "producer_npz": str(output_path),
        "valid_coordinate_fraction": float(np.mean(validity)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export MolmoMotion query trajectories as a producer NPZ and strict "
            "Causal4D external-forecast manifest."
        )
    )
    parser.add_argument("input_npz")
    parser.add_argument("output_npz")
    parser.add_argument("output_manifest_json")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--source-model", default="MolmoMotion")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-artifact-id")
    parser.add_argument(
        "--forecast",
        type=_forecast,
        action="append",
        required=True,
        help="repeat ID=CAPTION in the same order as future_positions K",
    )
    parser.add_argument(
        "--anchor-physical-frame",
        type=_nonnegative_integer,
        required=True,
    )
    parser.add_argument("--physical-fps", type=_positive_float, required=True)
    parser.add_argument("--forecast-fps", type=_positive_float, default=15.0)
    parser.add_argument(
        "--producer-environment",
        default="existing-molmomotion-environment",
    )
    parser.add_argument("--node-key", default="node_indices")
    parser.add_argument("--anchor-key", default="anchor_positions_world_m")
    parser.add_argument("--future-key", default="future_positions_world_m")
    parser.add_argument("--validity-key", default="validity_mask")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_bridge_package(
        args.input_npz,
        args.output_npz,
        args.output_manifest_json,
        case_id=args.case_id,
        source_model=args.source_model,
        source_revision=args.source_revision,
        source_artifact_id=args.source_artifact_id,
        forecasts=args.forecast,
        anchor_physical_frame=args.anchor_physical_frame,
        physical_fps=args.physical_fps,
        forecast_fps=args.forecast_fps,
        producer_environment=args.producer_environment,
        node_key=args.node_key,
        anchor_key=args.anchor_key,
        future_key=args.future_key,
        validity_key=args.validity_key,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
