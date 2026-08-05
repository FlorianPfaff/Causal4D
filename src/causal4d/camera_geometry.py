"""Validated pinhole-camera and rigid-transform primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

from causal4d.immutable_array import readonly_array


_DEFAULT_RIGID_ATOL = 1e-5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_pinhole_intrinsics(
    values: Any,
    *,
    name: str = "camera intrinsics",
) -> np.ndarray:
    """Return an owned read-only pinhole calibration matrix.

    The function accepts finite ``3 x 3`` matrices with positive focal lengths
    and the standard homogeneous row ``[0, 0, 1]``. Principal points and skew
    remain unconstrained because cropping and calibrated skew are valid inputs.
    """

    intrinsics = readonly_array(values, dtype=np.float64)
    _require(intrinsics.shape == (3, 3), f"{name} must have shape (3, 3)")
    _require(np.all(np.isfinite(intrinsics)), f"{name} must be finite")
    _require(
        intrinsics[0, 0] > 0.0 and intrinsics[1, 1] > 0.0,
        f"{name} focal lengths must be positive",
    )
    _require(
        np.allclose(
            intrinsics[2],
            np.asarray([0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=1e-12,
        ),
        f"{name} must use homogeneous row [0, 0, 1]",
    )
    return intrinsics


def validate_se3_transform(
    values: Any,
    *,
    name: str = "rigid transform",
    atol: float = _DEFAULT_RIGID_ATOL,
) -> np.ndarray:
    """Return an owned read-only homogeneous transform in ``SE(3)``.

    Scale, shear, reflections, malformed homogeneous rows, and non-finite values
    are rejected. The tolerance applies to rotation orthogonality and unit
    determinant checks.
    """

    _require(np.isfinite(atol) and atol > 0.0, "SE(3) tolerance must be positive")
    transform = readonly_array(values, dtype=np.float64)
    _require(transform.shape == (4, 4), f"{name} must have shape (4, 4)")
    _require(np.all(np.isfinite(transform)), f"{name} must be finite")
    _require(
        np.allclose(
            transform[3],
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=atol,
        ),
        f"{name} must use homogeneous row [0, 0, 0, 1]",
    )
    rotation = transform[:3, :3]
    _require(
        np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            rtol=0.0,
            atol=atol,
        ),
        f"{name} rotation must be orthonormal",
    )
    determinant = float(np.linalg.det(rotation))
    _require(
        np.isclose(determinant, 1.0, rtol=0.0, atol=5.0 * atol),
        f"{name} rotation must have determinant +1",
    )
    return transform


def invert_se3_transform(
    values: Any,
    *,
    name: str = "rigid transform",
    atol: float = _DEFAULT_RIGID_ATOL,
) -> np.ndarray:
    """Validate and invert an ``SE(3)`` transform analytically."""

    transform = validate_se3_transform(values, name=name, atol=atol)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return readonly_array(inverse)


__all__ = [
    "invert_se3_transform",
    "validate_pinhole_intrinsics",
    "validate_se3_transform",
]
