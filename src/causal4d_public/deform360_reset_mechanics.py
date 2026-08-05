"""Source-only reset-and-roll mechanics diagnostic for Deform360."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np


RESET_MECHANICS_SCHEMA_VERSION = 1
RESET_MECHANICS_KIND = "Deform360SourceResetMechanicsDiagnostic"
RESET_MECHANICS_CONFIG_KIND = "Deform360SourceResetMechanicsConfig"
SOURCE_MILESTONE = Path("milestones/deform360-replication-source-backend-v1")
_RESET_TECHNICAL_EXCEPTIONS = (
    ValueError,
    RuntimeError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_mapping(value: Any, *, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(message)
    return cast(Mapping[str, Any], value)


def _require_nonempty_list(value: Any, *, message: str) -> list[Any]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(message)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, *, name: str) -> float:
    _require(
        type(value) in {int, float} and type(value) is not bool and np.isfinite(value),
        f"{name} must be a finite number",
    )
    return float(value)


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    _require(result >= 0.0, f"{name} must be nonnegative")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    _require(type(value) is int and value >= 1, f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ResetMechanicsConfig:
    """Predeclared reset selection and mechanics-competence gates."""

    reset_count: int = 3
    horizon_observation_counts: tuple[int, ...] = (1, 3, 6)
    baseline_chamfer_tolerance_m: float = 5e-4
    baseline_strain_tolerance: float = 1e-2
    minimum_relative_improvement: float = 0.05
    minimum_win_fraction: float = 0.60
    minimum_common_episode_count: int = 24
    minimum_quality_valid_fraction: float = 0.90

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reset_count",
            _positive_integer(self.reset_count, name="reset_count"),
        )
        raw_horizons = self.horizon_observation_counts
        _require(
            isinstance(raw_horizons, tuple)
            and len(raw_horizons) >= 1
            and all(type(value) is int and value >= 1 for value in raw_horizons),
            "horizon_observation_counts must be a nonempty tuple of positive integers",
        )
        _require(
            tuple(sorted(set(raw_horizons))) == raw_horizons,
            "horizon_observation_counts must be strictly increasing",
        )
        for name in (
            "baseline_chamfer_tolerance_m",
            "baseline_strain_tolerance",
            "minimum_relative_improvement",
            "minimum_win_fraction",
            "minimum_quality_valid_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name=name),
            )
        _require(
            self.minimum_relative_improvement < 1.0,
            "minimum_relative_improvement must be below one",
        )
        _require(
            self.minimum_win_fraction <= 1.0,
            "minimum_win_fraction must be at most one",
        )
        _require(
            self.minimum_quality_valid_fraction <= 1.0,
            "minimum_quality_valid_fraction must be at most one",
        )
        object.__setattr__(
            self,
            "minimum_common_episode_count",
            _positive_integer(
                self.minimum_common_episode_count,
                name="minimum_common_episode_count",
            ),
        )

    @property
    def maximum_horizon_observation_count(self) -> int:
        return self.horizon_observation_counts[-1]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["horizon_observation_counts"] = list(self.horizon_observation_counts)
        return payload


def reset_mechanics_config_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a reset-mechanics lock without its self-reported digest."""

    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def load_reset_mechanics_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable source-only reset diagnostic lock."""

    lock_path = Path(path).resolve()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "reset-mechanics schema changed")
    _require(
        payload.get("artifact_kind") == RESET_MECHANICS_CONFIG_KIND,
        "reset-mechanics config kind changed",
    )
    _require(
        payload.get("config_sha256") == reset_mechanics_config_sha256(payload),
        "reset-mechanics config checksum mismatch",
    )
    controls = _require_mapping(
        payload.get("config"),
        message="reset-mechanics controls are missing",
    )
    selected = _require_nonempty_list(
        controls.get("selected_object_ids"),
        message="reset-mechanics object set is invalid",
    )
    _require(
        all(type(value) is str and value for value in selected)
        and len(selected) == len(set(selected)),
        "reset-mechanics object set is invalid",
    )
    _require(
        controls.get("candidate_selection")
        == "quality_constrained_source_oracle_else_finite_oracle",
        "reset-mechanics candidate-selection policy changed",
    )
    _require(
        controls.get("reset_selection")
        == "availability_only_evenly_spaced_including_prefix_and_latest_eligible",
        "reset-mechanics reset-selection policy changed",
    )
    _require(
        controls.get("initial_velocity_policy") == "zero_v1",
        "reset-mechanics initial-velocity policy changed",
    )
    boundary = _require_mapping(
        payload.get("information_boundary"),
        message="reset-mechanics information boundary is missing",
    )
    _require(
        boundary.get("source_only") is True
        and boundary.get("source_future_geometry_allowed_for_scoring") is True
        and boundary.get("source_tactile_allowed") is True
        and boundary.get("calibration_outcomes_allowed") is False
        and boundary.get("target_prefix_allowed") is False
        and boundary.get("target_future_allowed") is False
        and boundary.get("registered_replication_result_mutable") is False,
        "reset-mechanics lock opens a forbidden information or claim boundary",
    )
    config = ResetMechanicsConfig(
        reset_count=controls["reset_count"],
        horizon_observation_counts=tuple(controls["horizon_observation_counts"]),
        baseline_chamfer_tolerance_m=controls["baseline_chamfer_tolerance_m"],
        baseline_strain_tolerance=controls["baseline_strain_tolerance"],
        minimum_relative_improvement=controls["minimum_relative_improvement"],
        minimum_win_fraction=controls["minimum_win_fraction"],
        minimum_common_episode_count=controls["minimum_common_episode_count"],
        minimum_quality_valid_fraction=controls["minimum_quality_valid_fraction"],
    )
    return {
        "path": lock_path,
        "payload": payload,
        "config": config,
        "selected_object_ids": tuple(selected),
    }


def select_reset_positions(
    raw_hull_frame_indices: Sequence[int],
    *,
    reset_count: int,
    maximum_horizon_observation_count: int,
) -> tuple[int, ...]:
    """Select reset ordinals using frame availability only, never outcomes."""

    frames = np.asarray(raw_hull_frame_indices)
    count = _positive_integer(reset_count, name="reset_count")
    horizon = _positive_integer(
        maximum_horizon_observation_count,
        name="maximum_horizon_observation_count",
    )
    _require(frames.ndim == 1, "raw hull frames must be one-dimensional")
    _require(
        np.issubdtype(frames.dtype, np.integer) and len(frames) >= horizon + count,
        "raw hull frames do not support the reset ladder",
    )
    frames = frames.astype(np.int64, copy=False)
    _require(
        np.all(frames >= 0) and np.all(np.diff(frames) > 0),
        "raw hull frames must be nonnegative and strictly increasing",
    )
    latest_eligible = len(frames) - horizon - 1
    _require(
        latest_eligible >= count - 1,
        "not enough eligible reset positions for the configured ladder",
    )
    if count == 1:
        return (0,)
    positions = tuple(
        (index * latest_eligible) // (count - 1) for index in range(count)
    )
    _require(
        len(set(positions)) == count
        and positions[0] == 0
        and positions[-1] == latest_eligible,
        "reset selection did not produce the registered availability ladder",
    )
    return positions


def _horizon_key(horizon_observations: int) -> str:
    return f"next_{horizon_observations}_observations"


def _episode_has_technical_failure(episode: Mapping[str, Any]) -> bool:
    resets = episode.get("resets")
    if not isinstance(resets, list):
        return False
    return any(
        isinstance(reset, Mapping) and reset.get("status") == "technical_failure"
        for reset in resets
    )


def _episode_horizon_record(
    episode: Mapping[str, Any],
    horizon_observations: int,
) -> dict[str, Any] | None:
    resets = episode.get("resets")
    if not isinstance(resets, list) or len(resets) == 0:
        return None
    key = _horizon_key(horizon_observations)
    candidate_scores: list[float] = []
    persistence_scores: list[float] = []
    quality_flags: list[bool] = []
    for raw_reset in resets:
        reset = _require_mapping(raw_reset, message="reset record is not a mapping")
        if reset.get("status") != "completed":
            return None
        horizons = _require_mapping(
            reset.get("horizons"),
            message="reset horizon records are missing",
        )
        row = _require_mapping(
            horizons.get(key),
            message=f"reset horizon {key} is missing",
        )
        if row.get("finite") is not True:
            return None
        candidate_scores.append(
            _finite_float(row["mean_chamfer_m"], name="mean_chamfer_m")
        )
        persistence_scores.append(
            _finite_float(
                row["persistence_mean_chamfer_m"],
                name="persistence_mean_chamfer_m",
            )
        )
        quality_flags.append(row.get("quality_valid") is True)
    candidate = float(np.mean(candidate_scores))
    persistence = float(np.mean(persistence_scores))
    _require(persistence > 0.0, "episode persistence score must be positive")
    return {
        "episode_id": str(episode["episode_id"]),
        "object_id": str(episode["object_id"]),
        "reset_count": len(resets),
        "mean_chamfer_m": candidate,
        "persistence_mean_chamfer_m": persistence,
        "relative_improvement_vs_persistence": (persistence - candidate) / persistence,
        "win_vs_persistence": candidate < persistence,
        "quality_valid_fraction": float(np.mean(quality_flags)),
    }


def summarize_reset_horizon(
    episode_records: Sequence[Mapping[str, Any]],
    horizon_observations: int,
    *,
    config: ResetMechanicsConfig,
) -> dict[str, Any]:
    """Aggregate one horizon with the episode, not reset, as statistical unit."""

    horizon = _positive_integer(
        horizon_observations,
        name="horizon_observations",
    )
    _require(
        horizon in config.horizon_observation_counts,
        "horizon is not registered by the reset-mechanics config",
    )
    rows = [
        row
        for episode in episode_records
        if (row := _episode_horizon_record(episode, horizon)) is not None
    ]
    technical_failure_episode_count = sum(
        _episode_has_technical_failure(episode) for episode in episode_records
    )
    excluded_episode_count = len(episode_records) - len(rows)
    if not rows:
        return {
            "horizon_observations": horizon,
            "common_episode_count": 0,
            "excluded_episode_count": excluded_episode_count,
            "technical_failure_episode_count": technical_failure_episode_count,
            "mean_chamfer_m": None,
            "persistence_mean_chamfer_m": None,
            "relative_improvement_vs_persistence": None,
            "episode_win_fraction": None,
            "mean_quality_valid_fraction": None,
            "minimum_relative_improvement": config.minimum_relative_improvement,
            "minimum_win_fraction": config.minimum_win_fraction,
            "minimum_common_episode_count": config.minimum_common_episode_count,
            "minimum_quality_valid_fraction": (config.minimum_quality_valid_fraction),
            "passed": False,
            "episode_records": [],
        }
    candidate = float(np.mean([row["mean_chamfer_m"] for row in rows]))
    persistence = float(np.mean([row["persistence_mean_chamfer_m"] for row in rows]))
    _require(persistence > 0.0, "aggregate persistence score must be positive")
    improvement = (persistence - candidate) / persistence
    wins = float(np.mean([row["win_vs_persistence"] for row in rows]))
    quality = float(np.mean([row["quality_valid_fraction"] for row in rows]))
    passed = bool(
        len(rows) >= config.minimum_common_episode_count
        and improvement >= config.minimum_relative_improvement
        and wins >= config.minimum_win_fraction
        and quality >= config.minimum_quality_valid_fraction
    )
    return {
        "horizon_observations": horizon,
        "common_episode_count": len(rows),
        "excluded_episode_count": excluded_episode_count,
        "technical_failure_episode_count": technical_failure_episode_count,
        "mean_chamfer_m": candidate,
        "persistence_mean_chamfer_m": persistence,
        "relative_improvement_vs_persistence": improvement,
        "episode_win_fraction": wins,
        "mean_quality_valid_fraction": quality,
        "minimum_relative_improvement": config.minimum_relative_improvement,
        "minimum_win_fraction": config.minimum_win_fraction,
        "minimum_common_episode_count": config.minimum_common_episode_count,
        "minimum_quality_valid_fraction": config.minimum_quality_valid_fraction,
        "passed": passed,
        "episode_records": rows,
    }


def build_reset_mechanics_decision(
    episode_records: Sequence[Mapping[str, Any]],
    *,
    config: ResetMechanicsConfig,
) -> dict[str, Any]:
    """Apply the predeclared reset-and-roll competence ladder."""

    reproduction = [
        bool(record.get("prefix_baseline_reproduction", {}).get("passed"))
        for record in episode_records
    ]
    summaries = {
        _horizon_key(horizon): summarize_reset_horizon(
            episode_records,
            horizon,
            config=config,
        )
        for horizon in config.horizon_observation_counts
    }
    first_failure = next(
        (
            horizon
            for horizon in config.horizon_observation_counts
            if summaries[_horizon_key(horizon)]["passed"] is not True
        ),
        None,
    )
    technical_failure_episode_count = sum(
        _episode_has_technical_failure(record) for record in episode_records
    )
    technical_failure_reset_count = sum(
        1
        for record in episode_records
        for reset in record.get("resets", [])
        if isinstance(reset, Mapping) and reset.get("status") == "technical_failure"
    )
    baseline_passed = bool(reproduction and all(reproduction))
    passed = bool(baseline_passed and first_failure is None)
    first_summary = (
        summaries[_horizon_key(first_failure)] if first_failure is not None else None
    )
    if not baseline_passed:
        classification = "baseline_reproduction_failure"
        interpretation = (
            "the reset diagnostic cannot be interpreted because the frozen "
            "prefix baseline did not reproduce"
        )
    elif (
        first_summary is not None
        and first_summary["common_episode_count"] < config.minimum_common_episode_count
    ):
        classification = "insufficient_common_episode_support"
        interpretation = (
            "retained technical or nonfinite reset failures leave fewer complete "
            "episodes than the registered gate requires; no mechanics conclusion "
            "is permitted"
        )
    elif first_failure == config.horizon_observation_counts[0]:
        classification = "instantaneous_mechanics_or_contact_realization_failure"
        interpretation = (
            "observed-state resets do not rescue the first registered forecast "
            "horizon; prioritize contact realization, support, mass, or force laws"
        )
    elif first_failure is not None:
        classification = "multi_step_dynamics_accumulation_failure"
        interpretation = (
            "observed-state resets pass shorter horizons but fail by "
            f"{first_failure} observations; prioritize damping, integration, and "
            "topology-specific dynamics"
        )
    else:
        classification = "observed_reset_mechanics_competence_supported"
        interpretation = (
            "the current backend passes the source-only observed-reset ladder; "
            "the remaining prefix failure is more consistent with initialization, "
            "state estimation, or contact-state inference"
        )
    return {
        "baseline_reproduction_passed": baseline_passed,
        "baseline_reproduction_episode_count": len(reproduction),
        "technical_failure_episode_count": technical_failure_episode_count,
        "technical_failure_reset_count": technical_failure_reset_count,
        "horizon_summaries": summaries,
        "first_failed_horizon_observations": first_failure,
        "classification": classification,
        "passed": passed,
        "interpretation": interpretation,
        "registered_method_changed": False,
        "target_prefix_access_permitted": False,
        "target_future_access_permitted": False,
    }


def _clear_optional_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - integration environment
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_reset_prediction(
    observation: Any,
    prediction: np.ndarray,
    *,
    maximum_p99_relative_edge_strain: float,
    horizons: Sequence[int],
) -> dict[str, Any]:
    from .deform360_replication_warp import (
        sparse_graph_strain_summary,
        sparse_trajectory_chamfer_m,
    )

    forecast = np.asarray(prediction, dtype=np.float64)
    relative = (
        observation.raw_hull_frame_indices[1:] - observation.prefix_endpoint_frame
    )
    result: dict[str, Any] = {}
    for horizon in horizons:
        _require(
            horizon <= len(relative),
            "reset observation does not reach a registered horizon",
        )
        selected_relative = relative[:horizon]
        selected_reference = observation.reference_hulls_m[1 : horizon + 1]
        selected_prediction = forecast[selected_relative]
        candidate = sparse_trajectory_chamfer_m(
            selected_reference,
            selected_prediction,
        )
        persistence_prediction = np.repeat(
            observation.case.graph.positions_m[None],
            horizon,
            axis=0,
        )
        persistence = sparse_trajectory_chamfer_m(
            selected_reference,
            persistence_prediction,
        )
        strain = sparse_graph_strain_summary(
            observation.case.graph,
            forecast[: int(selected_relative[-1]) + 1],
        )
        mean = float(candidate["mean_m"])
        persistence_mean = float(persistence["mean_m"])
        p99 = float(strain["p99"])
        maximum = float(strain["maximum"])
        finite = bool(
            np.isfinite(mean)
            and np.isfinite(persistence_mean)
            and persistence_mean > 0.0
            and np.isfinite(p99)
            and np.isfinite(maximum)
        )
        result[_horizon_key(horizon)] = {
            "horizon_observations": horizon,
            "horizon_raw_frame_gap": int(selected_relative[-1]),
            "horizon_seconds": float(
                selected_relative[-1] * observation.case.dt_seconds
            ),
            "finite": finite,
            "mean_chamfer_m": mean if np.isfinite(mean) else None,
            "persistence_mean_chamfer_m": (
                persistence_mean if np.isfinite(persistence_mean) else None
            ),
            "relative_improvement_vs_persistence": (
                (persistence_mean - mean) / persistence_mean if finite else None
            ),
            "win_vs_persistence": bool(finite and mean < persistence_mean),
            "p99_relative_edge_strain": p99 if np.isfinite(p99) else None,
            "maximum_relative_edge_strain": (maximum if np.isfinite(maximum) else None),
            "quality_valid": bool(finite and p99 <= maximum_p99_relative_edge_strain),
        }
    return result


def _run_reset(
    observation: Any,
    official_phystwin_repo: Path,
    simulation_config: Any,
    candidate: Any,
    *,
    device: str,
    horizons: Sequence[int],
) -> dict[str, Any]:
    from .deform360_replication_case import score_replication_warp_prediction
    from .deform360_replication_warp import (
        OfficialWarpSparseGraphRunner,
        sparse_graph_strain_summary,
    )

    runner = OfficialWarpSparseGraphRunner(
        official_phystwin_repo,
        observation.case,
        simulation_config,
        device=device,
    )
    prediction = runner.rollout(candidate)
    full = score_replication_warp_prediction(observation, prediction)
    full_strain = sparse_graph_strain_summary(observation.case.graph, prediction)
    horizons_result = _score_reset_prediction(
        observation,
        prediction,
        maximum_p99_relative_edge_strain=(
            simulation_config.maximum_p99_relative_edge_strain
        ),
        horizons=horizons,
    )
    result = {
        "horizons": horizons_result,
        "full_remainder": {
            "future_observation_count": len(observation.reference_hulls_m) - 1,
            "mean_chamfer_m": (
                float(full["mean_m"]) if np.isfinite(full["mean_m"]) else None
            ),
            "late_chamfer_m": (
                float(full["late_mean_m"]) if np.isfinite(full["late_mean_m"]) else None
            ),
            "p99_relative_edge_strain": (
                float(full_strain["p99"]) if np.isfinite(full_strain["p99"]) else None
            ),
            "maximum_relative_edge_strain": (
                float(full_strain["maximum"])
                if np.isfinite(full_strain["maximum"])
                else None
            ),
        },
    }
    del runner, prediction
    _clear_optional_cuda_cache()
    return result


def _evaluate_registered_reset(
    *,
    build_observation: Any,
    episode_dir: Path,
    episode_id: str,
    stratum: str,
    frames: np.ndarray,
    hulls: Sequence[np.ndarray],
    schedule: Any,
    reset_ordinal: int,
    reset_position: int,
    official_phystwin_repo: Path,
    simulation_config: Any,
    candidate: Any,
    device: str,
    horizons: Sequence[int],
) -> dict[str, Any]:
    base = {
        "reset_ordinal": reset_ordinal,
        "reset_hull_position": reset_position,
        "reset_raw_frame": int(frames[reset_position]),
        "available_future_observation_count": len(frames) - reset_position - 1,
    }
    stage = "build_observation"
    try:
        observation = build_observation(
            episode_dir,
            episode_id,
            stratum,
            frames[reset_position:],
            hulls[reset_position:],
            schedule,
        )
        stage = "rollout_and_score"
        evaluation = _run_reset(
            observation,
            official_phystwin_repo,
            simulation_config,
            candidate,
            device=device,
            horizons=horizons,
        )
    except _RESET_TECHNICAL_EXCEPTIONS as exc:
        _clear_optional_cuda_cache()
        return {
            **base,
            "status": "technical_failure",
            "technical_failure": {
                "stage": stage,
                "exception_type": type(exc).__name__,
                "message": str(exc) or repr(exc),
            },
        }
    return {
        **base,
        "status": "completed",
        "contact_associations": list(observation.contact_associations),
        **evaluation,
    }


def _episode_record(
    *,
    repository_root: Path,
    data_root: Path,
    source_grid_path: Path,
    cohort: Mapping[str, Any],
    official_phystwin_repo: Path,
    device: str,
    config: ResetMechanicsConfig,
) -> dict[str, Any]:
    from .deform360_phystwin_feasibility import (
        WarpRopeCandidate,
        WarpRopeFeasibilityConfig,
    )
    from .deform360_prefix_kinematics_diagnostic import (
        select_fixed_source_candidate,
    )
    from .deform360_replication_case import build_replication_warp_observation
    from .deform360_replication_contact import (
        ReplicationOpeningContactModel,
        contact_state_by_robot_axis,
        load_replication_contact_episode,
    )
    from .deform360_replication_fit import validate_source_warp_candidate_grid
    from .deform360_replication_geometry import load_replication_hull_archive

    grid = json.loads(source_grid_path.read_text(encoding="utf-8"))
    validate_source_warp_candidate_grid(grid)
    selected = select_fixed_source_candidate(grid)
    object_id = str(cohort["object_id"])
    episode_id = str(grid["episode_id"])
    _require(
        episode_id.startswith(f"{object_id}/episode_"),
        "source grid and cohort object differ",
    )
    raw_episode_index = episode_id.rsplit("episode_", maxsplit=1)[1]
    _require(
        len(raw_episode_index) == 4 and raw_episode_index.isdigit(),
        "source episode identity is malformed",
    )
    episode_index = int(raw_episode_index)
    _require(
        episode_index in cohort["source_episode_ids"],
        "reset diagnostic was given a non-source episode",
    )
    metadata = cohort["episodes"][str(episode_index)]
    episode_dir = data_root / "aligned" / object_id / f"episode_{episode_index:04d}"
    episode = load_replication_contact_episode(
        episode_dir,
        episode_id=episode_id,
        bimanual=metadata["bimanual"] == "yes",
        nonprehensile=metadata["nonprehensile"] == "yes",
    )
    contact_model = ReplicationOpeningContactModel(**grid["contact_model"])
    schedule = contact_state_by_robot_axis(
        episode,
        contact_model.tactile_group_to_robot_axis,
    )
    hull_json = (
        data_root
        / "observations"
        / object_id
        / f"episode_{episode_index:04d}"
        / "sampled_hulls.json"
    )
    hull_payload = json.loads(hull_json.read_text(encoding="utf-8"))
    _require(
        hull_payload["result_sha256"] == grid["reference_geometry_result_sha256"],
        "source hull identity changed",
    )
    frames, hulls = load_replication_hull_archive(hull_payload)
    total_frame_count = len(hulls)
    available = np.asarray([len(hull) > 0 for hull in hulls], dtype=bool)
    frames = frames[available]
    hulls = tuple(hull for hull, keep in zip(hulls, available, strict=True) if keep)
    _require(
        total_frame_count == grid["reference_geometry_total_frame_count"]
        and len(hulls) == grid["reference_geometry_available_frame_count"],
        "source hull availability changed",
    )
    _require(
        frames.astype(int).tolist() == grid["raw_hull_frame_indices"],
        "source hull frame indices changed",
    )
    reset_positions = select_reset_positions(
        frames,
        reset_count=config.reset_count,
        maximum_horizon_observation_count=(config.maximum_horizon_observation_count),
    )
    simulation_config = WarpRopeFeasibilityConfig(**grid["config"])
    candidate = WarpRopeCandidate(**selected["parameters"])
    resets = [
        _evaluate_registered_reset(
            build_observation=build_replication_warp_observation,
            episode_dir=episode_dir,
            episode_id=episode_id,
            stratum=str(grid["stratum"]),
            frames=frames,
            hulls=hulls,
            schedule=schedule,
            reset_ordinal=reset_ordinal,
            reset_position=reset_position,
            official_phystwin_repo=official_phystwin_repo,
            simulation_config=simulation_config,
            candidate=candidate,
            device=device,
            horizons=config.horizon_observation_counts,
        )
        for reset_ordinal, reset_position in enumerate(reset_positions)
    ]
    completed_reset_count = sum(reset["status"] == "completed" for reset in resets)
    technical_failure_reset_count = len(resets) - completed_reset_count
    prefix = (
        resets[0].get("full_remainder") if resets[0]["status"] == "completed" else None
    )
    mean_delta = (
        abs(float(prefix["mean_chamfer_m"]) - selected["archived_mean_chamfer_m"])
        if isinstance(prefix, Mapping) and prefix["mean_chamfer_m"] is not None
        else None
    )
    strain_delta = (
        abs(
            float(prefix["p99_relative_edge_strain"])
            - selected["archived_p99_relative_edge_strain"]
        )
        if isinstance(prefix, Mapping)
        and prefix["p99_relative_edge_strain"] is not None
        else None
    )
    reproduction_passed = bool(
        mean_delta is not None
        and strain_delta is not None
        and mean_delta <= config.baseline_chamfer_tolerance_m
        and strain_delta <= config.baseline_strain_tolerance
    )
    return {
        "object_id": object_id,
        "stratum": str(grid["stratum"]),
        "episode_id": episode_id,
        "source_grid_path": str(source_grid_path.relative_to(repository_root)),
        "source_grid_result_sha256": grid["result_sha256"],
        "selected_candidate": selected,
        "raw_hull_frame_indices": frames.astype(int).tolist(),
        "reset_positions": list(reset_positions),
        "resets": resets,
        "completed_reset_count": completed_reset_count,
        "technical_failure_reset_count": technical_failure_reset_count,
        "technically_complete": technical_failure_reset_count == 0,
        "prefix_baseline_reproduction": {
            "mean_chamfer_absolute_delta_m": mean_delta,
            "p99_strain_absolute_delta": strain_delta,
            "maximum_mean_chamfer_absolute_delta_m": (
                config.baseline_chamfer_tolerance_m
            ),
            "maximum_p99_strain_absolute_delta": config.baseline_strain_tolerance,
            "passed": reproduction_passed,
        },
        "information_boundary": {
            "source_episode_only": True,
            "reset_selection_uses_availability_only": True,
            "source_future_geometry_read_for_scoring": True,
            "source_future_tactile_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
        },
    }


def run_source_reset_mechanics_diagnostic(
    repository_root: str | Path,
    protocol_path: str | Path,
    data_root: str | Path,
    official_phystwin_repo: str | Path,
    output_path: str | Path,
    *,
    lock_path: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run the locked observed-reset mechanics-competence ladder."""

    from causal4d.atomic_io import atomic_write_json

    from .deform360_prefix_kinematics_diagnostic import verify_source_milestone
    from .deform360_replication import load_deform360_replication_protocol

    lock = load_reset_mechanics_lock(lock_path)
    config: ResetMechanicsConfig = lock["config"]
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    data = Path(data_root).resolve()
    official = Path(official_phystwin_repo).resolve()
    _require(root.is_dir(), "repository root is missing")
    _require(protocol_file.is_file(), "replication protocol is missing")
    _require(data.is_dir(), "Deform360 derived-data root is missing")
    _require(official.is_dir(), "official PhysTwin repository is missing")
    milestone_verification = verify_source_milestone(root)
    protocol = load_deform360_replication_protocol(protocol_file)
    _require(
        lock["payload"]["protocol_config_sha256"] == protocol["config_sha256"],
        "locked replication protocol identity changed",
    )
    _require(
        lock["payload"]["source_milestone_manifest_sha256"]
        == milestone_verification["manifest_sha256"],
        "locked source milestone identity changed",
    )
    decision_path = root / SOURCE_MILESTONE / "artifacts/source_backend_decision.json"
    source_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    _require(
        lock["payload"]["source_backend_decision_result_sha256"]
        == source_decision["result_sha256"],
        "locked source-backend decision identity changed",
    )
    cohorts = {
        str(record["object_id"]): record for record in protocol["config"]["cohort"]
    }
    grid_root = root / SOURCE_MILESTONE / "artifacts/source-grids"
    records = []
    for object_id in lock["selected_object_ids"]:
        _require(object_id in cohorts, f"locked object is absent: {object_id}")
        paths = sorted((grid_root / object_id).glob("source_episode_*_grid.json"))
        expected = len(cohorts[object_id]["source_episode_ids"])
        _require(
            len(paths) == expected,
            f"{object_id} source-grid set is incomplete",
        )
        for path in paths:
            records.append(
                _episode_record(
                    repository_root=root,
                    data_root=data,
                    source_grid_path=path,
                    cohort=cohorts[object_id],
                    official_phystwin_repo=official,
                    device=device,
                    config=config,
                )
            )
    decision = build_reset_mechanics_decision(records, config=config)
    payload = {
        "schema_version": RESET_MECHANICS_SCHEMA_VERSION,
        "artifact_kind": RESET_MECHANICS_KIND,
        "config": config.as_dict(),
        "protocol": {
            "path": (
                str(protocol_file.relative_to(root))
                if protocol_file.is_relative_to(root)
                else str(protocol_file)
            ),
            "sha256": _sha256_file(protocol_file),
            "config_sha256": protocol["config_sha256"],
        },
        "source_milestone": milestone_verification,
        "diagnostic_lock": {
            "path": (
                str(lock["path"].relative_to(root))
                if lock["path"].is_relative_to(root)
                else str(lock["path"])
            ),
            "file_sha256": _sha256_file(lock["path"]),
            "config_sha256": lock["payload"]["config_sha256"],
        },
        "selected_object_ids": list(lock["selected_object_ids"]),
        "episode_records": records,
        "decision": decision,
        "information_boundary": {
            "source_candidate_outcomes_read": True,
            "source_future_geometry_read_for_scoring": True,
            "source_future_tactile_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
            "registered_replication_result_changed": False,
            "registered_36_execution_method_changed": False,
        },
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    atomic_write_json(output_path, payload)
    return payload


def validate_source_reset_mechanics_diagnostic(payload: Mapping[str, Any]) -> None:
    """Validate result identity, reset ladder, and target-closed boundary."""

    _require(
        payload.get("schema_version") == RESET_MECHANICS_SCHEMA_VERSION,
        "reset-mechanics diagnostic schema changed",
    )
    _require(
        payload.get("artifact_kind") == RESET_MECHANICS_KIND,
        "reset-mechanics diagnostic kind changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "reset-mechanics diagnostic checksum mismatch",
    )
    config_mapping = _require_mapping(
        payload.get("config"),
        message="reset-mechanics result config is missing",
    )
    config = ResetMechanicsConfig(
        reset_count=config_mapping["reset_count"],
        horizon_observation_counts=tuple(config_mapping["horizon_observation_counts"]),
        baseline_chamfer_tolerance_m=config_mapping["baseline_chamfer_tolerance_m"],
        baseline_strain_tolerance=config_mapping["baseline_strain_tolerance"],
        minimum_relative_improvement=config_mapping["minimum_relative_improvement"],
        minimum_win_fraction=config_mapping["minimum_win_fraction"],
        minimum_common_episode_count=config_mapping["minimum_common_episode_count"],
        minimum_quality_valid_fraction=config_mapping["minimum_quality_valid_fraction"],
    )
    boundary = _require_mapping(
        payload.get("information_boundary"),
        message="reset-mechanics result boundary is missing",
    )
    _require(
        boundary.get("calibration_outcomes_read") is False
        and boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False
        and boundary.get("registered_replication_result_changed") is False
        and boundary.get("registered_36_execution_method_changed") is False,
        "reset-mechanics diagnostic crossed its information or claim boundary",
    )
    records = _require_nonempty_list(
        payload.get("episode_records"),
        message="reset-mechanics diagnostic has no episodes",
    )
    episode_ids: set[str] = set()
    for raw_record in records:
        record = _require_mapping(
            raw_record,
            message="reset-mechanics episode is not a mapping",
        )
        raw_episode_id = record.get("episode_id")
        if type(raw_episode_id) is not str or not raw_episode_id:
            raise ValueError("reset-mechanics episode identity is invalid or repeated")
        episode_id = raw_episode_id
        _require(
            episode_id not in episode_ids,
            "reset-mechanics episode identity is invalid or repeated",
        )
        episode_ids.add(episode_id)
        raw_frames = record.get("raw_hull_frame_indices")
        if not isinstance(raw_frames, list) or any(
            type(frame) is not int for frame in raw_frames
        ):
            raise ValueError("reset-mechanics episode omitted raw hull frames")
        frames = cast(list[int], raw_frames)
        expected_positions = select_reset_positions(
            frames,
            reset_count=config.reset_count,
            maximum_horizon_observation_count=(
                config.maximum_horizon_observation_count
            ),
        )
        _require(
            record.get("reset_positions") == list(expected_positions),
            "reset-mechanics episode changed its availability-only reset ladder",
        )
        resets = _require_nonempty_list(
            record.get("resets"),
            message="reset-mechanics episode has no resets",
        )
        _require(
            len(resets) == config.reset_count,
            "reset-mechanics episode reset count changed",
        )
        completed_reset_count = 0
        technical_failure_reset_count = 0
        for reset_ordinal, raw_reset in enumerate(resets):
            reset = _require_mapping(
                raw_reset,
                message="reset-mechanics reset is not a mapping",
            )
            _require(
                reset.get("reset_ordinal") == reset_ordinal
                and reset.get("reset_hull_position")
                == expected_positions[reset_ordinal],
                "reset-mechanics reset ordering changed",
            )
            status = reset.get("status")
            _require(
                status in {"completed", "technical_failure"},
                "reset-mechanics reset status is invalid",
            )
            if status == "completed":
                completed_reset_count += 1
                _require(
                    "technical_failure" not in reset,
                    "completed reset contains technical-failure metadata",
                )
                horizons = _require_mapping(
                    reset.get("horizons"),
                    message="reset-mechanics reset horizons are missing",
                )
                _require(
                    tuple(horizons)
                    == tuple(
                        _horizon_key(horizon)
                        for horizon in config.horizon_observation_counts
                    ),
                    "reset-mechanics horizon set or ordering changed",
                )
            else:
                technical_failure_reset_count += 1
                _require(
                    "horizons" not in reset and "full_remainder" not in reset,
                    "technical-failure reset contains scientific scores",
                )
                failure = _require_mapping(
                    reset.get("technical_failure"),
                    message="reset technical-failure metadata is missing",
                )
                _require(
                    set(failure) == {"stage", "exception_type", "message"}
                    and all(
                        type(failure[field]) is str and failure[field]
                        for field in ("stage", "exception_type", "message")
                    ),
                    "reset technical-failure metadata is invalid",
                )
        _require(
            record.get("completed_reset_count") == completed_reset_count
            and record.get("technical_failure_reset_count")
            == technical_failure_reset_count
            and record.get("technically_complete")
            is (technical_failure_reset_count == 0),
            "reset-mechanics episode technical-failure accounting changed",
        )
        episode_boundary = _require_mapping(
            record.get("information_boundary"),
            message="reset-mechanics episode boundary is missing",
        )
        _require(
            episode_boundary.get("source_episode_only") is True
            and episode_boundary.get("reset_selection_uses_availability_only") is True
            and episode_boundary.get("calibration_outcomes_read") is False
            and episode_boundary.get("target_prefix_read") is False
            and episode_boundary.get("target_future_read") is False,
            "reset-mechanics episode crossed its source-only boundary",
        )


__all__ = [
    "RESET_MECHANICS_CONFIG_KIND",
    "RESET_MECHANICS_KIND",
    "RESET_MECHANICS_SCHEMA_VERSION",
    "ResetMechanicsConfig",
    "build_reset_mechanics_decision",
    "load_reset_mechanics_lock",
    "reset_mechanics_config_sha256",
    "run_source_reset_mechanics_diagnostic",
    "select_reset_positions",
    "summarize_reset_horizon",
    "validate_source_reset_mechanics_diagnostic",
]
