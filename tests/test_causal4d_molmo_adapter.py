import numpy as np

import causal4d.molmo_adapter as molmo_adapter
from causal4d.molmo_adapter import (
    camera_to_world_points,
    farthest_point_indices,
    prepare_molmo_phystwin_query,
)


def test_farthest_points_are_deterministic_and_distributed() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [1.5, 0.0]])
    first = farthest_point_indices(points, 3)
    second = farthest_point_indices(points, 3)
    assert np.array_equal(first, second)
    assert {0, 3} <= set(first.tolist())


def test_camera_forecast_transforms_to_world_coordinates() -> None:
    points = np.asarray([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]])
    transform = np.eye(4)
    transform[:3, 3] = [0.5, -1.0, 2.0]
    world = camera_to_world_points(points, transform)
    assert np.allclose(world, points + np.asarray([0.5, -1.0, 2.0]))


def test_prepare_query_hash_locks_legacy_inputs(tmp_path, monkeypatch) -> None:
    final_path = tmp_path / "processed" / "final_data.pkl"
    final_path.parent.mkdir()
    final_path.write_bytes(b"placeholder")
    raw_path = tmp_path / "raw"
    (raw_path / "color" / "0").mkdir(parents=True)
    for frame in (0, 2):
        (raw_path / "color" / "0" / f"{frame}.png").write_bytes(b"image")
    (raw_path / "metadata.json").write_text(
        '{"fps": 2, "intrinsics": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], "WH": [10, 10]}',
        encoding="utf-8",
    )

    final_digest = "1" * 64
    calibration_digest = "2" * 64
    final_data = {
        "object_points": np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ]
        ),
        "object_visibilities": np.ones((4, 2), dtype=bool),
        "object_motions_valid": np.ones((4, 2), dtype=bool),
    }
    calibration = np.eye(4)[None]
    calls = []

    def load_legacy(path, *, expected_sha256, artifact_kind, required_keys=()):
        calls.append((str(path), expected_sha256, artifact_kind, tuple(required_keys)))
        if str(path).endswith("final_data.pkl"):
            return final_data
        assert str(path).endswith("calibrate.pkl")
        return calibration

    class Mapping:
        track_paths = (raw_path / "cotracker" / "0.npz",)
        tracks_by_camera = (
            np.asarray(
                [
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[1.0, 1.0], [2.0, 2.0]],
                ]
            ),
        )
        visibility_by_camera = (np.ones((4, 2), dtype=bool),)
        source_camera = np.asarray((0, 0))
        source_track = np.asarray((0, 1))

    def load_raw_map(final_data_path, raw_case_dir, *, final_data_sha256):
        assert final_data_path == final_path
        assert raw_case_dir == raw_path
        assert final_data_sha256 == final_digest
        return Mapping()

    monkeypatch.setattr(
        molmo_adapter,
        "load_trusted_legacy_phystwin_pickle",
        load_legacy,
    )
    monkeypatch.setattr(
        molmo_adapter,
        "load_released_phystwin_raw_track_map",
        load_raw_map,
    )

    query = prepare_molmo_phystwin_query(
        final_path,
        raw_path,
        final_data_sha256=final_digest,
        calibration_sha256=calibration_digest,
        train_end_frame=3,
        history_size=2,
        point_count=1,
        camera_index=0,
        forecast_fps=1.0,
    )

    assert query.final_data_sha256 == final_digest
    assert query.calibration_sha256 == calibration_digest
    assert calls[0][1:] == (
        final_digest,
        "mapping",
        ("object_points", "object_visibilities", "object_motions_valid"),
    )
    assert calls[1][1:] == (calibration_digest, "ndarray", ())
