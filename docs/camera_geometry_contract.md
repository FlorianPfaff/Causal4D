# Camera Geometry Contract

Causal4D's public multiview utilities treat camera calibration as an explicit
input contract. A matrix with the right shape is not automatically a valid
camera model.

## Pinhole intrinsics

`causal4d.camera_geometry.validate_pinhole_intrinsics` requires:

- a finite `3 x 3` matrix;
- positive horizontal and vertical focal lengths; and
- the homogeneous row `[0, 0, 1]`.

The principal point and skew are retained as calibrated. The validator returns
an owned read-only array so later caller-side mutation cannot change an active
projection calculation.

## Rigid extrinsics

`validate_se3_transform` requires a finite homogeneous `4 x 4` transform with:

- final row `[0, 0, 0, 1]`;
- an orthonormal rotation block; and
- rotation determinant `+1`.

Scale, shear, and reflections are rejected rather than accepted by a generic
matrix inverse. `invert_se3_transform` validates the input and uses the exact
rigid inverse

```text
R_inv = R^T

t_inv = -R^T t
```

instead of treating the calibration as an arbitrary affine matrix.

## Consumers

The contract is applied before projection in:

- the source-only SAM2 cross-view reliability audit;
- adaptive multiview visual-hull carving; and
- thin-rope Gaussian-splat geometry diagnostics.

Malformed calibration fails before a reliability score, hull, or splat-quality
result is emitted. This is input and numerical hardening only. It does not
change a frozen scientific method or provide new reconstruction, calibration,
or physical-prediction evidence.
