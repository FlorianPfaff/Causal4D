"""Causal prefix-kinematic initialization for Deform360 sparse graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_replication_graph import Deform360SparseGraph


PREFIX_KINEMATICS_POLICY_VERSION = 1
PREFIX_KINEMATICS_POLICIES = (
    "zero_v1",
    "global_contact_translation_v1",
    "graph_harmonic_contact_v1",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_finite_float(value: Any, *, name: str) -> float:
    _require(
        type(value) in {int, float}
        and type(value) is not bool
        and np.isfinite(value)
        and float(value) > 0.0,
        f"{name} must be a positive finite number",
    )
    return float(value)


def _positive_integer(value: Any, *, name: str) -> int:
    _require(type(value) is int and value >= 1, f"{name} must be a positive integer")
    return value


def _readonly(values: np.ndarray, *, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PrefixKinematicsConfig:
    """Fixed controls for source-only object-velocity initialization."""

    lookback_frames: int = 3
    contact_memory_frames: int = 3
    contact_weight: float = 100.0
    smoothness_weight: float = 1.0
    ridge_weight: float = 0.1
    maximum_controller_speed_m_s: float = 2.0
    maximum_node_speed_m_s: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lookback_frames",
            _positive_integer(self.lookback_frames, name="lookback_frames"),
        )
        object.__setattr__(
            self,
            "contact_memory_frames",
            _positive_integer(
                self.contact_memory_frames,
                name="contact_memory_frames",
            ),
        )
        for name in (
            "contact_weight",
            "smoothness_weight",
            "ridge_weight",
            "maximum_controller_speed_m_s",
            "maximum_node_speed_m_s",
        ):
            object.__setattr__(
                self,
                name,
                _positive_finite_float(getattr(self, name), name=name),
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def clip_velocity_norms(values_m_s: np.ndarray, maximum_speed_m_s: float) -> np.ndarray:
    """Radially clip finite three-dimensional velocities."""

    values = np.asarray(values_m_s, dtype=np.float64)
    maximum = _positive_finite_float(
        maximum_speed_m_s,
        name="maximum_speed_m_s",
    )
    _require(
        values.ndim == 2 and values.shape[1] == 3,
        "velocities must have shape (N,3)",
    )
    _require(np.all(np.isfinite(values)), "velocities must be finite")
    speed = np.linalg.norm(values, axis=1)
    scale = np.ones_like(speed)
    moving = speed > maximum
    scale[moving] = maximum / speed[moving]
    return _readonly(values * scale[:, None], dtype=np.float64)


def recent_contact_mask(
    contact_active: np.ndarray,
    *,
    prefix_endpoint_frame: int,
    memory_frames: int,
) -> np.ndarray:
    """Return controllers active at least once in the causal prefix tail."""

    active = np.asarray(contact_active, dtype=bool)
    _require(active.ndim == 2 and active.shape[1] >= 1, "contact state must be (T,C)")
    _require(
        type(prefix_endpoint_frame) is int
        and 0 <= prefix_endpoint_frame < len(active),
        "prefix endpoint is outside the contact schedule",
    )
    memory = _positive_integer(memory_frames, name="memory_frames")
    start = max(0, prefix_endpoint_frame - memory + 1)
    return _readonly(
        np.any(active[start : prefix_endpoint_frame + 1], axis=0),
        dtype=bool,
    )


def _robot_prefix_arrays(
    state: Any,
    *,
    frame_stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    _require(type(frame_stop) is int and frame_stop >= 1, "frame_stop must be positive")
    openings = np.asarray(state.openings[:frame_stop], dtype=np.float64)
    transforms = np.asarray(state.T_worlds[:frame_stop], dtype=np.float64)
    if openings.ndim == 1:
        openings = openings[:, None]
    if transforms.ndim == 3:
        transforms = transforms[:, None]
    _require(openings.ndim == 2, "robot openings must have shape (T,C)")
    _require(
        transforms.shape == (*openings.shape, 4, 4),
        "robot transforms must have shape (T,C,4,4)",
    )
    _require(
        np.all(np.isfinite(openings)) and np.all(np.isfinite(transforms)),
        "robot prefix contains nonfinite values",
    )
    return openings, transforms


def _association_values(
    association: Mapping[str, Any],
    *,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    _require(
        type(association.get("robot_axis")) is int
        and int(association["robot_axis"]) == axis,
        "contact association robot axis changed",
    )
    raw_indices = association.get("selected_taxel_indices")
    _require(
        isinstance(raw_indices, Sequence)
        and not isinstance(raw_indices, (str, bytes))
        and len(raw_indices) >= 1,
        "contact association has no taxel indices",
    )
    _require(
        all(type(index) is int and index >= 0 for index in raw_indices),
        "contact association taxel indices must be nonnegative integers",
    )
    indices = np.asarray(raw_indices, dtype=np.int64)
    _require(
        len(np.unique(indices)) == len(indices),
        "contact association repeats a taxel index",
    )
    offset = np.asarray(association.get("contact_offset_m"), dtype=np.float64)
    _require(
        offset.shape == (3,) and np.all(np.isfinite(offset)),
        "contact association offset must be a finite three-vector",
    )
    return indices, offset


def controller_patch_velocities_from_prefix(
    episode_dir: str | Path,
    *,
    prefix_endpoint_frame: int,
    contact_associations: Sequence[Mapping[str, Any]],
    dt_seconds: float,
    lookback_frames: int,
    maximum_speed_m_s: float,
) -> np.ndarray:
    """Estimate gripper-patch velocities from frames no later than the prefix end."""

    dt = _positive_finite_float(dt_seconds, name="dt_seconds")
    lookback = _positive_integer(lookback_frames, name="lookback_frames")
    _require(
        type(prefix_endpoint_frame) is int and prefix_endpoint_frame >= lookback,
        "prefix endpoint does not contain the configured lookback",
    )
    _require(
        isinstance(contact_associations, Sequence)
        and not isinstance(contact_associations, (str, bytes))
        and len(contact_associations) >= 1,
        "contact associations must be a nonempty sequence",
    )
    try:
        from deform360.processing.control_points_stage import gripper_taxel_points
        from deform360.robot import load_robot_state
    except ImportError as error:  # pragma: no cover - host integration
        raise RuntimeError("the pinned Deform360 runtime is required") from error

    directory = Path(episode_dir).resolve()
    state = load_robot_state(directory / "robot" / "robot.npz")
    openings, transforms = _robot_prefix_arrays(
        state,
        frame_stop=prefix_endpoint_frame + 1,
    )
    _require(
        prefix_endpoint_frame < len(openings),
        "prefix endpoint is outside the robot trajectory",
    )
    controller_count = openings.shape[1]
    _require(
        len(contact_associations) == controller_count,
        "contact-association count differs from robot axes",
    )
    start = prefix_endpoint_frame - lookback
    velocities = np.empty((controller_count, 3), dtype=np.float64)
    for axis, association in enumerate(contact_associations):
        indices, offset = _association_values(association, axis=axis)
        patches = []
        for frame in (start, prefix_endpoint_frame):
            taxels = np.asarray(
                gripper_taxel_points(
                    float(openings[frame, axis]),
                    transforms[frame, axis],
                ),
                dtype=np.float64,
            )
            _require(
                taxels.ndim == 2
                and taxels.shape[1] == 3
                and len(taxels) > int(np.max(indices))
                and np.all(np.isfinite(taxels)),
                "gripper taxel geometry is incompatible with the association",
            )
            patches.append(np.mean(taxels[indices], axis=0) + offset)
        velocities[axis] = (patches[1] - patches[0]) / (lookback * dt)
    return clip_velocity_norms(velocities, maximum_speed_m_s)


def global_contact_translation_velocity(
    graph: Deform360SparseGraph,
    controller_velocities_m_s: np.ndarray,
    active_controllers: np.ndarray,
    *,
    maximum_node_speed_m_s: float,
) -> np.ndarray:
    """Apply the mean recent-contact velocity as one rigid translation field."""

    controller = np.asarray(controller_velocities_m_s, dtype=np.float64)
    active = np.asarray(active_controllers, dtype=bool)
    _require(
        controller.ndim == 2 and controller.shape[1] == 3,
        "controller velocities must have shape (C,3)",
    )
    _require(active.shape == (len(controller),), "active-controller shape differs")
    _require(np.all(np.isfinite(controller)), "controller velocities must be finite")
    if not np.any(active):
        return _readonly(np.zeros_like(graph.positions_m), dtype=np.float64)
    mean = np.mean(controller[active], axis=0)
    field = np.repeat(mean[None], len(graph.positions_m), axis=0)
    return clip_velocity_norms(field, maximum_node_speed_m_s)


def graph_harmonic_contact_velocity(
    graph: Deform360SparseGraph,
    controller_velocities_m_s: np.ndarray,
    active_controllers: np.ndarray,
    contact_node_indices: Sequence[int],
    *,
    contact_weight: float,
    smoothness_weight: float,
    ridge_weight: float,
    maximum_node_speed_m_s: float,
) -> np.ndarray:
    """Diffuse recent contact velocities through the stretch/shear graph."""

    controller = np.asarray(controller_velocities_m_s, dtype=np.float64)
    active = np.asarray(active_controllers, dtype=bool)
    _require(
        controller.ndim == 2 and controller.shape[1] == 3,
        "controller velocities must have shape (C,3)",
    )
    _require(active.shape == (len(controller),), "active-controller shape differs")
    _require(np.all(np.isfinite(controller)), "controller velocities must be finite")
    _require(
        isinstance(contact_node_indices, Sequence)
        and not isinstance(contact_node_indices, (str, bytes))
        and len(contact_node_indices) == len(controller),
        "contact-node count differs from controllers",
    )
    node_count = len(graph.positions_m)
    _require(
        all(
            type(node) is int and 0 <= node < node_count
            for node in contact_node_indices
        ),
        "contact node is outside the graph",
    )
    contact = _positive_finite_float(contact_weight, name="contact_weight")
    smoothness = _positive_finite_float(
        smoothness_weight,
        name="smoothness_weight",
    )
    ridge = _positive_finite_float(ridge_weight, name="ridge_weight")
    if not np.any(active):
        return _readonly(np.zeros_like(graph.positions_m), dtype=np.float64)

    stretch = graph.spring_edges[graph.spring_families == 0]
    _require(len(stretch) >= node_count - 1, "stretch/shear graph is disconnected")
    lengths = np.linalg.norm(
        graph.positions_m[stretch[:, 1]] - graph.positions_m[stretch[:, 0]],
        axis=1,
    )
    _require(np.all(lengths > 0.0), "stretch/shear graph contains a zero edge")
    scale = float(np.median(lengths))
    weights = scale / lengths
    system = ridge * np.eye(node_count, dtype=np.float64)
    right = np.zeros((node_count, 3), dtype=np.float64)
    for (left, right_node), edge_weight in zip(stretch, weights, strict=True):
        value = smoothness * float(edge_weight)
        system[left, left] += value
        system[right_node, right_node] += value
        system[left, right_node] -= value
        system[right_node, left] -= value
    for controller_index, enabled in enumerate(active):
        if not enabled:
            continue
        node = int(contact_node_indices[controller_index])
        system[node, node] += contact
        right[node] += contact * controller[controller_index]
    field = np.linalg.solve(system, right)
    _require(
        np.all(np.isfinite(field)),
        "graph velocity solve produced nonfinite values",
    )
    return clip_velocity_norms(field, maximum_node_speed_m_s)


def build_prefix_velocity_policies(
    graph: Deform360SparseGraph,
    episode_dir: str | Path,
    *,
    prefix_endpoint_frame: int,
    contact_associations: Sequence[Mapping[str, Any]],
    full_contact_active: np.ndarray,
    contact_node_indices: Sequence[int],
    dt_seconds: float,
    config: PrefixKinematicsConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct fixed zero, rigid, and graph-harmonic prefix velocity fields."""

    cfg = config or PrefixKinematicsConfig()
    controller = controller_patch_velocities_from_prefix(
        episode_dir,
        prefix_endpoint_frame=prefix_endpoint_frame,
        contact_associations=contact_associations,
        dt_seconds=dt_seconds,
        lookback_frames=cfg.lookback_frames,
        maximum_speed_m_s=cfg.maximum_controller_speed_m_s,
    )
    active = recent_contact_mask(
        full_contact_active,
        prefix_endpoint_frame=prefix_endpoint_frame,
        memory_frames=cfg.contact_memory_frames,
    )
    zero = _readonly(np.zeros_like(graph.positions_m), dtype=np.float64)
    global_field = global_contact_translation_velocity(
        graph,
        controller,
        active,
        maximum_node_speed_m_s=cfg.maximum_node_speed_m_s,
    )
    graph_field = graph_harmonic_contact_velocity(
        graph,
        controller,
        active,
        contact_node_indices,
        contact_weight=cfg.contact_weight,
        smoothness_weight=cfg.smoothness_weight,
        ridge_weight=cfg.ridge_weight,
        maximum_node_speed_m_s=cfg.maximum_node_speed_m_s,
    )
    policies = {
        PREFIX_KINEMATICS_POLICIES[0]: zero,
        PREFIX_KINEMATICS_POLICIES[1]: global_field,
        PREFIX_KINEMATICS_POLICIES[2]: graph_field,
    }

    def summary(values: np.ndarray) -> dict[str, float]:
        speed = np.linalg.norm(values, axis=1)
        return {
            "mean_speed_m_s": float(np.mean(speed)),
            "maximum_speed_m_s": float(np.max(speed)),
            "rms_speed_m_s": float(np.sqrt(np.mean(np.square(speed)))),
        }

    diagnostics = {
        "policy_version": PREFIX_KINEMATICS_POLICY_VERSION,
        "config": cfg.as_dict(),
        "prefix_endpoint_frame": prefix_endpoint_frame,
        "causal_start_frame": prefix_endpoint_frame - cfg.lookback_frames,
        "active_controller_indices": np.flatnonzero(active).astype(int).tolist(),
        "controller_velocities_m_s": controller.tolist(),
        "policy_velocity_summary": {
            name: summary(values) for name, values in policies.items()
        },
        "future_robot_state_read": False,
        "future_object_geometry_read": False,
        "future_tactile_read": False,
    }
    return policies, diagnostics


__all__ = [
    "PREFIX_KINEMATICS_POLICIES",
    "PREFIX_KINEMATICS_POLICY_VERSION",
    "PrefixKinematicsConfig",
    "build_prefix_velocity_policies",
    "clip_velocity_norms",
    "controller_patch_velocities_from_prefix",
    "global_contact_translation_velocity",
    "graph_harmonic_contact_velocity",
    "recent_contact_mask",
]
