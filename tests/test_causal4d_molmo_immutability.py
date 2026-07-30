from pathlib import Path

import numpy as np
import pytest

from causal4d.molmo_adapter import MolmoForecastBundle, MolmoPhysTwinQuery


def _query(tmp_path: Path) -> tuple[MolmoPhysTwinQuery, dict[str, np.ndarray]]:
    image_paths = []
    for frame in (0, 2, 4):
        image_path = tmp_path / f"{frame}.png"
        image_path.write_bytes(b"image")
        image_paths.append(image_path)

    inputs = {
        "history": np.asarray([0, 2, 4], dtype=int),
        "nodes": np.asarray([1, 3], dtype=int),
        "raw_tracks": np.asarray([7, 9], dtype=int),
        "points_2d": np.asarray([[10.0, 20.0], [30.0, 40.0]]),
        "points_3d": np.arange(18, dtype=float).reshape(3, 2, 3),
        "camera_to_world": np.eye(4),
        "intrinsics": np.eye(3),
    }
    query = MolmoPhysTwinQuery(
        case_name="synthetic",
        raw_case_dir=tmp_path,
        camera_index=0,
        t0_frame=4,
        history_frame_indices=inputs["history"],
        image_paths=tuple(image_paths),
        node_indices=inputs["nodes"],
        raw_track_indices=inputs["raw_tracks"],
        points_2d_xy=inputs["points_2d"],
        points_3d_world_history_m=inputs["points_3d"],
        camera_to_world=inputs["camera_to_world"],
        intrinsics=inputs["intrinsics"],
        source_fps=30.0,
        forecast_fps=15.0,
        frame_stride=2,
    )
    return query, inputs


def test_molmo_query_arrays_are_owned_and_read_only(tmp_path: Path) -> None:
    query, inputs = _query(tmp_path)
    expected = {
        "history": query.history_frame_indices.copy(),
        "nodes": query.node_indices.copy(),
        "raw_tracks": query.raw_track_indices.copy(),
        "points_2d": query.points_2d_xy.copy(),
        "points_3d": query.points_3d_world_history_m.copy(),
        "camera_to_world": query.camera_to_world.copy(),
        "intrinsics": query.intrinsics.copy(),
    }

    for values in inputs.values():
        values[...] = -123

    pairs = (
        (query.history_frame_indices, expected["history"]),
        (query.node_indices, expected["nodes"]),
        (query.raw_track_indices, expected["raw_tracks"]),
        (query.points_2d_xy, expected["points_2d"]),
        (query.points_3d_world_history_m, expected["points_3d"]),
        (query.camera_to_world, expected["camera_to_world"]),
        (query.intrinsics, expected["intrinsics"]),
    )
    for actual, frozen in pairs:
        np.testing.assert_array_equal(actual, frozen)
        assert not actual.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            actual[...] = 0


def test_molmo_forecast_arrays_are_owned_and_read_only(tmp_path: Path) -> None:
    query, _ = _query(tmp_path)
    camera = np.arange(36, dtype=float).reshape(2, 2, 3, 3)
    world = camera + 100.0
    expected_camera = camera.copy()
    expected_world = world.copy()

    bundle = MolmoForecastBundle(
        query=query,
        forecast_ids=("lift", "lower"),
        captions=("Lift it.", "Lower it."),
        future_camera_m=camera,
        future_world_m=world,
        raw_text=("<tracks>lift", "<tracks>lower"),
        checkpoint="synthetic",
    )
    camera[...] = -1.0
    world[...] = -2.0

    np.testing.assert_array_equal(bundle.future_camera_m, expected_camera)
    np.testing.assert_array_equal(bundle.future_world_m, expected_world)
    assert not bundle.future_camera_m.flags.writeable
    assert not bundle.future_world_m.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        bundle.future_camera_m[...] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        bundle.future_world_m[...] = 0.0
