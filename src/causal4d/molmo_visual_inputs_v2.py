"""Digest-bound PhysTwin visual query preparation for MolmoMotion."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.molmo_adapter import (
    MolmoForecastBundle,
    MolmoPhysTwinQuery,
    farthest_point_indices,
    run_molmo_motion_forecasts,
)

VISUAL_INPUT_MANIFEST_SCHEMA = "causal4d.phystwin_visual_inputs"
VISUAL_INPUT_MANIFEST_VERSION = 2
BPT_ARTIFACT_PROVIDER_API = "bayesian_phystwin.causal4d_artifacts_v2"


def _digest(value: str, *, name: str) -> str:
    normalized = str(value)
    if (
        normalized != normalized.lower()
        or len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected_sha256: str, *, name: str) -> None:
    expected = _digest(expected_sha256, name=name)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not hmac.compare_digest(_sha256(path), expected):
        raise ValueError(f"{name} mismatch for {path}")


def _digest_items(value: object, *, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty object")
    items = tuple(
        sorted(
            (
                str(path),
                _digest(str(digest), name=f"{name}[{path!r}]"),
            )
            for path, digest in value.items()
        )
    )
    if any(
        not path or Path(path).is_absolute() or ".." in Path(path).parts
        for path, _ in items
    ):
        raise ValueError(f"{name} paths must be safe relative paths")
    return items


@dataclass(frozen=True)
class MolmoVisualInputManifestV2:
    """Frozen mapping, calibration, and selected RGB identities for one query."""

    final_data_sha256: str
    metadata_sha256: str
    pcd_sha256: str
    calibration_sha256: str
    cotracker_sha256: tuple[tuple[str, str], ...]
    image_sha256: tuple[tuple[str, str], ...]
    initial_match_tolerance_m: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "final_data_sha256",
            _digest(self.final_data_sha256, name="final_data_sha256"),
        )
        object.__setattr__(
            self,
            "metadata_sha256",
            _digest(self.metadata_sha256, name="metadata_sha256"),
        )
        object.__setattr__(
            self,
            "pcd_sha256",
            _digest(self.pcd_sha256, name="pcd_sha256"),
        )
        object.__setattr__(
            self,
            "calibration_sha256",
            _digest(self.calibration_sha256, name="calibration_sha256"),
        )
        cotracker = tuple(
            (str(path), _digest(value, name=f"cotracker_sha256[{path!r}]"))
            for path, value in self.cotracker_sha256
        )
        images = tuple(
            (str(path), _digest(value, name=f"image_sha256[{path!r}]"))
            for path, value in self.image_sha256
        )
        for name, items, prefix in (
            ("cotracker_sha256", cotracker, "cotracker/"),
            ("image_sha256", images, "color/"),
        ):
            if not items or len({path for path, _ in items}) != len(items):
                raise ValueError(f"{name} must contain unique nonempty paths")
            if any(
                not path.startswith(prefix)
                or Path(path).is_absolute()
                or ".." in Path(path).parts
                for path, _ in items
            ):
                raise ValueError(f"{name} contains an invalid released path")
        if (
            not np.isfinite(self.initial_match_tolerance_m)
            or self.initial_match_tolerance_m <= 0.0
        ):
            raise ValueError("initial_match_tolerance_m must be positive and finite")
        object.__setattr__(self, "cotracker_sha256", tuple(sorted(cotracker)))
        object.__setattr__(self, "image_sha256", tuple(sorted(images)))

    @classmethod
    def load(cls, path: str | Path) -> MolmoVisualInputManifestV2:
        """Load and validate the exact manifest schema used by the V2 CLI."""

        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("visual input manifest must contain an object")
        expected_fields = {
            "schema_name",
            "schema_version",
            "provider_api",
            "final_data_sha256",
            "metadata_sha256",
            "pcd_sha256",
            "calibration_sha256",
            "cotracker_sha256",
            "image_sha256",
            "initial_match_tolerance_m",
        }
        actual_fields = set(map(str, payload))
        if actual_fields != expected_fields:
            raise ValueError(
                "visual input manifest fields changed: "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"extra={sorted(actual_fields - expected_fields)}"
            )
        if payload["schema_name"] != VISUAL_INPUT_MANIFEST_SCHEMA:
            raise ValueError("unsupported visual input manifest schema")
        if payload["schema_version"] != VISUAL_INPUT_MANIFEST_VERSION:
            raise ValueError("unsupported visual input manifest version")
        if payload["provider_api"] != BPT_ARTIFACT_PROVIDER_API:
            raise ValueError("unexpected Bayesian-PhysTwin artifact provider API")
        return cls(
            final_data_sha256=str(payload["final_data_sha256"]),
            metadata_sha256=str(payload["metadata_sha256"]),
            pcd_sha256=str(payload["pcd_sha256"]),
            calibration_sha256=str(payload["calibration_sha256"]),
            cotracker_sha256=_digest_items(
                payload["cotracker_sha256"], name="cotracker_sha256"
            ),
            image_sha256=_digest_items(payload["image_sha256"], name="image_sha256"),
            initial_match_tolerance_m=float(payload["initial_match_tolerance_m"]),
        )


@dataclass(frozen=True)
class MolmoPhysTwinQueryV2(MolmoPhysTwinQuery):
    """Molmo query extended with complete mapping and selected-image provenance."""

    visual_input_artifact_id: str = ""
    visual_input_manifest_sha256: str = ""
    image_sha256: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        artifact_id = _digest(
            self.visual_input_artifact_id,
            name="visual_input_artifact_id",
        )
        manifest_digest = _digest(
            self.visual_input_manifest_sha256,
            name="visual_input_manifest_sha256",
        )
        images = tuple(
            (str(path), _digest(value, name=f"image_sha256[{path!r}]"))
            for path, value in self.image_sha256
        )
        if not images or len({path for path, _ in images}) != len(images):
            raise ValueError("image_sha256 must identify unique history images")
        object.__setattr__(self, "visual_input_artifact_id", artifact_id)
        object.__setattr__(self, "visual_input_manifest_sha256", manifest_digest)
        object.__setattr__(self, "image_sha256", tuple(sorted(images)))

    def metadata(self) -> dict[str, Any]:
        values = super().metadata()
        values.update(
            {
                "visual_input_contract": VISUAL_INPUT_MANIFEST_SCHEMA,
                "visual_input_contract_version": VISUAL_INPUT_MANIFEST_VERSION,
                "visual_input_artifact_id": self.visual_input_artifact_id,
                "visual_input_manifest_sha256": self.visual_input_manifest_sha256,
                "image_sha256": dict(self.image_sha256),
            }
        )
        return values


def _load_bpt_visual_inputs(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    manifest: MolmoVisualInputManifestV2,
):
    from bayesian_phystwin.causal4d_artifacts_v2 import (
        load_released_phystwin_visual_inputs,
    )

    return load_released_phystwin_visual_inputs(
        final_data_path,
        raw_case_dir,
        final_data_sha256=manifest.final_data_sha256,
        metadata_sha256=manifest.metadata_sha256,
        pcd_sha256=manifest.pcd_sha256,
        calibration_sha256=manifest.calibration_sha256,
        cotracker_sha256=dict(manifest.cotracker_sha256),
        initial_match_tolerance_m=manifest.initial_match_tolerance_m,
    )


def _verify_query_images(query: MolmoPhysTwinQueryV2) -> None:
    expected = dict(query.image_sha256)
    actual_names = tuple(
        path.relative_to(query.raw_case_dir).as_posix() for path in query.image_paths
    )
    if set(expected) != set(actual_names):
        raise ValueError("query history images differ from the frozen digest inventory")
    for name in actual_names:
        _verify(
            query.raw_case_dir / name,
            expected[name],
            name=f"image_sha256[{name!r}]",
        )


def prepare_molmo_phystwin_query_v2(
    final_data_path: str | Path,
    raw_case_dir: str | Path,
    visual_input_manifest_path: str | Path,
    *,
    train_end_frame: int,
    history_size: int = 3,
    point_count: int = 8,
    camera_index: int | None = None,
    forecast_fps: float = 15.0,
) -> MolmoPhysTwinQueryV2:
    """Prepare a query using only the immutable V2 artifact returned by BPT."""

    manifest_path = Path(visual_input_manifest_path)
    manifest = MolmoVisualInputManifestV2.load(manifest_path)
    artifact = _load_bpt_visual_inputs(final_data_path, raw_case_dir, manifest)
    object_points = np.asarray(artifact.object_points_m, dtype=float)
    visible = np.asarray(artifact.object_visibility, dtype=bool)
    motion_valid = np.asarray(artifact.object_motion_valid, dtype=bool)
    frame_count, track_count, coordinate_count = object_points.shape
    if coordinate_count != 3 or visible.shape != (frame_count, track_count):
        raise ValueError("visual artifact object arrays have incompatible shapes")
    if motion_valid.shape != (frame_count, track_count):
        raise ValueError("visual artifact motion validity has an incompatible shape")
    if not history_size <= train_end_frame < frame_count:
        raise ValueError("train_end_frame cannot provide the requested history")
    if point_count < 1:
        raise ValueError("point_count must be positive")
    if not np.isfinite(forecast_fps) or forecast_fps <= 0.0:
        raise ValueError("forecast_fps must be finite and positive")
    ratio = artifact.source_fps / float(forecast_fps)
    frame_stride = int(round(ratio))
    if frame_stride < 1 or not np.isclose(ratio, frame_stride):
        raise ValueError("raw source fps must be an integer multiple of forecast_fps")
    t0 = train_end_frame - 1
    history_frames = np.arange(
        t0 - frame_stride * (history_size - 1),
        t0 + 1,
        frame_stride,
        dtype=int,
    )
    if history_frames[0] < 0:
        raise ValueError("train_end_frame cannot provide the requested sampled history")

    candidates_by_camera: list[np.ndarray] = []
    for camera in range(len(artifact.track_paths)):
        camera_nodes = np.flatnonzero(artifact.source_camera == camera)
        camera_raw_ids = artifact.source_track[camera_nodes]
        raw_visibility = artifact.visibility_by_camera[camera][
            history_frames[:, None], camera_raw_ids[None]
        ]
        raw_tracks = artifact.tracks_by_camera[camera][
            history_frames[:, None], camera_raw_ids[None]
        ]
        row = raw_tracks[..., 0]
        column = raw_tracks[..., 1]
        in_bounds = (
            (row >= 0.0)
            & (row < artifact.image_height)
            & (column >= 0.0)
            & (column < artifact.image_width)
        )
        processed_valid = np.all(
            visible[history_frames][:, camera_nodes]
            & motion_valid[history_frames][:, camera_nodes],
            axis=0,
        )
        eligible = (
            processed_valid
            & np.all(raw_visibility & in_bounds, axis=0)
            & np.all(
                np.isfinite(object_points[history_frames][:, camera_nodes]),
                axis=(0, 2),
            )
        )
        candidates_by_camera.append(camera_nodes[eligible])

    if camera_index is None:
        camera = int(np.argmax([len(values) for values in candidates_by_camera]))
    else:
        camera = int(camera_index)
        if not 0 <= camera < len(candidates_by_camera):
            raise ValueError("camera_index is unavailable")
    candidates = candidates_by_camera[camera]
    if len(candidates) < point_count:
        raise ValueError(
            f"camera {camera} has only {len(candidates)} valid tracks; need {point_count}"
        )
    selected_local = farthest_point_indices(object_points[t0, candidates], point_count)
    nodes = candidates[selected_local]
    raw_ids = artifact.source_track[nodes]
    raw_row_column = artifact.tracks_by_camera[camera][t0, raw_ids]
    points_xy = raw_row_column[:, [1, 0]]
    raw_path = Path(raw_case_dir)
    image_paths = tuple(
        raw_path / "color" / str(camera) / f"{frame}.png" for frame in history_frames
    )
    expected_images = dict(manifest.image_sha256)
    required_image_names = tuple(
        path.relative_to(raw_path).as_posix() for path in image_paths
    )
    if set(expected_images) != set(required_image_names):
        missing = sorted(set(required_image_names) - set(expected_images))
        extra = sorted(set(expected_images) - set(required_image_names))
        raise ValueError(
            f"history image digest inventory differs: missing={missing}, extra={extra}"
        )
    for name in required_image_names:
        _verify(
            raw_path / name,
            expected_images[name],
            name=f"image_sha256[{name!r}]",
        )

    query = MolmoPhysTwinQueryV2(
        case_name=Path(final_data_path).resolve().parent.name,
        raw_case_dir=raw_path,
        camera_index=camera,
        t0_frame=t0,
        history_frame_indices=history_frames,
        image_paths=image_paths,
        node_indices=nodes,
        raw_track_indices=raw_ids,
        points_2d_xy=points_xy,
        points_3d_world_history_m=object_points[history_frames][:, nodes],
        camera_to_world=artifact.camera_to_world[camera],
        intrinsics=artifact.intrinsics[camera],
        source_fps=float(artifact.source_fps),
        forecast_fps=float(forecast_fps),
        frame_stride=frame_stride,
        final_data_sha256=manifest.final_data_sha256,
        calibration_sha256=manifest.calibration_sha256,
        visual_input_artifact_id=artifact.artifact_id,
        visual_input_manifest_sha256=_sha256(manifest_path),
        image_sha256=manifest.image_sha256,
    )
    _verify_query_images(query)
    return query


def run_molmo_motion_forecasts_v2(
    query: MolmoPhysTwinQueryV2,
    checkpoint: str | Path,
    captions: Mapping[str, str],
    *,
    future_horizon: int = 30,
    device: str = "cuda",
) -> MolmoForecastBundle:
    """Revalidate selected RGB inputs immediately before and after inference."""

    _verify_query_images(query)
    result = run_molmo_motion_forecasts(
        query,
        checkpoint,
        captions,
        future_horizon=future_horizon,
        device=device,
    )
    _verify_query_images(query)
    return result


__all__ = [
    "BPT_ARTIFACT_PROVIDER_API",
    "MolmoPhysTwinQueryV2",
    "MolmoVisualInputManifestV2",
    "VISUAL_INPUT_MANIFEST_SCHEMA",
    "VISUAL_INPUT_MANIFEST_VERSION",
    "prepare_molmo_phystwin_query_v2",
    "run_molmo_motion_forecasts_v2",
]
