from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import causal4d.molmo_visual_inputs_v2 as visual_v2
from causal4d.molmo_visual_inputs_v2 import (
    MolmoPhysTwinQueryV2,
    MolmoVisualInputManifestV2,
    prepare_molmo_phystwin_query_v2,
    run_molmo_motion_forecasts_v2,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    final_data = tmp_path / "case" / "final_data.pkl"
    final_data.parent.mkdir(parents=True)
    final_data.write_bytes(b"final")
    raw_case = tmp_path / "raw"
    for frame in (0, 2):
        path = raw_case / "color" / "0" / f"{frame}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{frame}".encode())
    manifest = raw_case / "visual-inputs-v2.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_name": "causal4d.phystwin_visual_inputs",
                "schema_version": 2,
                "provider_api": "bayesian_phystwin.causal4d_artifacts_v2",
                "final_data_sha256": "a" * 64,
                "metadata_sha256": "b" * 64,
                "pcd_sha256": "c" * 64,
                "calibration_sha256": "d" * 64,
                "cotracker_sha256": {
                    "cotracker/camera0.npz": "e" * 64,
                    "cotracker/camera1.npz": "f" * 64,
                },
                "image_sha256": {
                    "color/0/0.png": _sha256(raw_case / "color/0/0.png"),
                    "color/0/2.png": _sha256(raw_case / "color/0/2.png"),
                },
                "initial_match_tolerance_m": 1e-6,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return final_data, raw_case, manifest


def _artifact(raw_case: Path):
    tracks_camera0 = np.zeros((4, 1, 2), dtype=float)
    tracks_camera1 = np.zeros((4, 2, 2), dtype=float)
    tracks_camera1[:, 1] = 1.0
    return SimpleNamespace(
        artifact_id="1" * 64,
        object_points_m=np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
                [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
                [[0.3, 0.0, 0.0], [1.3, 0.0, 0.0]],
            ]
        ),
        object_visibility=np.ones((4, 2), dtype=bool),
        object_motion_valid=np.ones((4, 2), dtype=bool),
        track_paths=(
            raw_case / "cotracker/camera0.npz",
            raw_case / "cotracker/camera1.npz",
        ),
        tracks_by_camera=(tracks_camera0, tracks_camera1),
        visibility_by_camera=(
            np.ones((4, 1), dtype=bool),
            np.ones((4, 2), dtype=bool),
        ),
        source_camera=np.asarray((0, 1)),
        source_track=np.asarray((0, 1)),
        intrinsics=np.repeat(np.eye(3)[None], 2, axis=0),
        camera_to_world=np.repeat(np.eye(4)[None], 2, axis=0),
        source_fps=30.0,
        image_width=8,
        image_height=8,
    )


def test_manifest_rejects_unknown_schema_and_unsafe_paths(tmp_path: Path) -> None:
    _, _, manifest_path = _write_inputs(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["schema_name"] = "unknown"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported visual input manifest"):
        MolmoVisualInputManifestV2.load(manifest_path)

    payload["schema_name"] = "causal4d.phystwin_visual_inputs"
    payload["image_sha256"] = {"../escape.png": "a" * 64}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="safe relative paths"):
        MolmoVisualInputManifestV2.load(manifest_path)


def test_prepare_v2_uses_camera_local_track_indices_and_binds_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    final_data, raw_case, manifest_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        visual_v2,
        "_load_bpt_visual_inputs",
        lambda final, raw, manifest: _artifact(raw_case),
    )

    query = prepare_molmo_phystwin_query_v2(
        final_data,
        raw_case,
        manifest_path,
        train_end_frame=3,
        history_size=2,
        point_count=1,
        camera_index=0,
        forecast_fps=15.0,
    )

    assert isinstance(query, MolmoPhysTwinQueryV2)
    np.testing.assert_array_equal(query.node_indices, (0,))
    np.testing.assert_array_equal(query.raw_track_indices, (0,))
    np.testing.assert_array_equal(query.history_frame_indices, (0, 2))
    assert query.visual_input_artifact_id == "1" * 64
    assert query.metadata()["image_sha256"] == dict(query.image_sha256)


def test_prepare_v2_rejects_missing_or_tampered_history_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    final_data, raw_case, manifest_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        visual_v2,
        "_load_bpt_visual_inputs",
        lambda final, raw, manifest: _artifact(raw_case),
    )
    payload = json.loads(manifest_path.read_text())
    payload["image_sha256"].pop("color/0/2.png")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="history image digest inventory differs"):
        prepare_molmo_phystwin_query_v2(
            final_data,
            raw_case,
            manifest_path,
            train_end_frame=3,
            history_size=2,
            point_count=1,
            camera_index=0,
        )

    _, _, manifest_path = _write_inputs(tmp_path / "second")
    second_raw = manifest_path.parent
    monkeypatch.setattr(
        visual_v2,
        "_load_bpt_visual_inputs",
        lambda final, raw, manifest: _artifact(second_raw),
    )
    (second_raw / "color/0/2.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="image_sha256"):
        prepare_molmo_phystwin_query_v2(
            manifest_path.parents[1] / "case/final_data.pkl",
            second_raw,
            manifest_path,
            train_end_frame=3,
            history_size=2,
            point_count=1,
            camera_index=0,
        )


def test_run_v2_revalidates_images_after_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    final_data, raw_case, manifest_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        visual_v2,
        "_load_bpt_visual_inputs",
        lambda final, raw, manifest: _artifact(raw_case),
    )
    query = prepare_molmo_phystwin_query_v2(
        final_data,
        raw_case,
        manifest_path,
        train_end_frame=3,
        history_size=2,
        point_count=1,
        camera_index=0,
    )

    def mutate_and_return(query_arg, checkpoint, captions, **kwargs):
        query_arg.image_paths[-1].write_bytes(b"changed-during-inference")
        return object()

    monkeypatch.setattr(visual_v2, "run_molmo_motion_forecasts", mutate_and_return)
    with pytest.raises(ValueError, match="image_sha256"):
        run_molmo_motion_forecasts_v2(
            query,
            tmp_path / "checkpoint",
            {"action": "move it"},
        )
