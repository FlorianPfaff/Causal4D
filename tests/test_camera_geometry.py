from __future__ import annotations

import numpy as np
import pytest

from causal4d.camera_geometry import (
    invert_se3_transform,
    validate_pinhole_intrinsics,
    validate_se3_transform,
)
from causal4d_public.deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    multiview_mask_consistency,
)
from causal4d_public.deform360_splat_probe import (
    ThinRopeSplatProbeConfig,
    gaussian_splat_geometry_diagnostics,
)
from causal4d_public.deform360_visual_hull import carve_candidate_points


def _camera_intrinsics() -> np.ndarray:
    return np.asarray(
        [
            [40.0, 0.0, 16.0],
            [0.0, 40.0, 16.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _camera_to_world() -> np.ndarray:
    angle = 0.37
    cosine = np.cos(angle)
    sine = np.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = np.asarray([0.1, -0.2, 0.3])
    return transform


def test_rigid_transform_is_owned_read_only_and_inverted_analytically() -> None:
    source = _camera_to_world()
    original = source.copy()

    validated = validate_se3_transform(source)
    inverse = invert_se3_transform(source)

    source[0, 3] = 99.0
    assert validated[0, 3] == pytest.approx(original[0, 3])
    assert np.allclose(inverse, np.linalg.inv(original), rtol=0.0, atol=1e-12)
    assert validated.flags.writeable is False
    assert inverse.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        validated[0, 0] = 2.0


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (np.diag([1.01, 1.0, 1.0, 1.0]), "orthonormal"),
        (np.diag([-1.0, 1.0, 1.0, 1.0]), "determinant"),
        (
            np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.1, 1.0],
                ]
            ),
            "homogeneous row",
        ),
    ],
)
def test_rigid_transform_rejects_non_se3_matrices(
    transform: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_se3_transform(transform)


def test_pinhole_intrinsics_reject_invalid_calibration() -> None:
    intrinsics = validate_pinhole_intrinsics(_camera_intrinsics())
    assert intrinsics.flags.writeable is False

    negative_focal = _camera_intrinsics()
    negative_focal[0, 0] = -1.0
    with pytest.raises(ValueError, match="focal lengths"):
        validate_pinhole_intrinsics(negative_focal)

    bad_row = _camera_intrinsics()
    bad_row[2, 0] = 0.1
    with pytest.raises(ValueError, match="homogeneous row"):
        validate_pinhole_intrinsics(bad_row)


def test_public_multiview_paths_reject_scaled_extrinsics() -> None:
    intrinsics = _camera_intrinsics()
    valid = np.eye(4)
    scaled = np.eye(4)
    scaled[0, 0] = 1.01

    masks3 = {
        "a": np.ones((32, 32), dtype=bool),
        "b": np.ones((32, 32), dtype=bool),
        "c": np.ones((32, 32), dtype=bool),
    }
    intrinsics3 = {camera: intrinsics for camera in masks3}
    extrinsics3 = {"a": valid, "b": valid, "c": scaled}
    with pytest.raises(ValueError, match="orthonormal"):
        multiview_mask_consistency(
            masks3,
            intrinsics3,
            extrinsics3,
            CrossViewMaskReliabilityConfig(
                voxel_resolution=16,
                minimum_consensus_votes=2,
                minimum_leave_one_out_recall=0.0,
            ),
        )

    points = np.column_stack(
        (
            np.linspace(-0.1, 0.1, 20),
            np.zeros(20),
            np.ones(20),
        )
    )
    masks2 = {"a": masks3["a"], "b": masks3["b"]}
    intrinsics2 = {"a": intrinsics, "b": intrinsics}
    extrinsics2 = {"a": valid, "b": scaled}
    with pytest.raises(ValueError, match="orthonormal"):
        carve_candidate_points(
            points,
            masks2,
            intrinsics2,
            extrinsics2,
            consensus_fraction_of_peak=0.5,
            minimum_consensus_votes=2,
        )

    with pytest.raises(ValueError, match="orthonormal"):
        gaussian_splat_geometry_diagnostics(
            points,
            opacity=np.ones(len(points)),
            masks_by_camera=masks2,
            intrinsics_by_camera=intrinsics2,
            camera_to_world_by_camera=extrinsics2,
            config=ThinRopeSplatProbeConfig(
                minimum_camera_count=2,
                minimum_gaussian_count=1,
            ),
        )
