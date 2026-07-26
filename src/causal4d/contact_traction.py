"""Low-rank graph traction and resultant-wrench utilities."""

from __future__ import annotations

import numpy as np


def graph_traction_field(
    graph_basis: np.ndarray,
    coefficient_forces_n: np.ndarray,
) -> np.ndarray:
    """Lift low-rank graph traction coefficients to per-node forces."""

    basis = np.asarray(graph_basis, dtype=float)
    coefficients = np.asarray(coefficient_forces_n, dtype=float)
    if basis.ndim != 2 or not np.all(np.isfinite(basis)):
        raise ValueError("graph_basis must be a finite matrix")
    if coefficients.ndim < 2 or coefficients.shape[-2:] != (basis.shape[1], 3):
        raise ValueError(
            "coefficient_forces_n must have trailing shape (graph_rank, 3)"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficient_forces_n must be finite")
    return np.einsum("nr,...rc->...nc", basis, coefficients)


def integrate_contact_wrench(
    node_positions_m: np.ndarray,
    node_forces_n: np.ndarray,
    contact_origin_m: np.ndarray,
) -> np.ndarray:
    """Integrate graph-node forces into ``[Fx,Fy,Fz,Tx,Ty,Tz]``."""

    positions = np.asarray(node_positions_m, dtype=float)
    forces = np.asarray(node_forces_n, dtype=float)
    if positions.ndim < 2 or positions.shape[-1] != 3:
        raise ValueError("node_positions_m must have trailing shape (node, 3)")
    try:
        positions, forces = np.broadcast_arrays(positions, forces)
    except ValueError as error:
        raise ValueError(
            "node positions and forces must be broadcast-compatible"
        ) from error
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(forces)):
        raise ValueError("node positions and forces must be finite")
    origin = np.asarray(contact_origin_m, dtype=float)
    expected_origin_shape = positions.shape[:-2] + (3,)
    try:
        origin = np.broadcast_to(origin, expected_origin_shape)
    except ValueError as error:
        raise ValueError(
            f"contact_origin_m must broadcast to {expected_origin_shape}"
        ) from error
    if not np.all(np.isfinite(origin)):
        raise ValueError("contact_origin_m must be finite")
    resultant_force = np.sum(forces, axis=-2)
    lever_arms = positions - origin[..., None, :]
    resultant_torque = np.sum(np.cross(lever_arms, forces), axis=-2)
    return np.concatenate((resultant_force, resultant_torque), axis=-1)
