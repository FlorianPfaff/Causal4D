"""Source-only Deform360 diagnostic for causal initial object velocity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.atomic_io import atomic_write_json

from .deform360_phystwin_feasibility import (
    WarpRopeCandidate,
    WarpRopeFeasibilityConfig,
)
from .deform360_prefix_kinematics import (
    PREFIX_KINEMATICS_POLICIES,
    PrefixKinematicsConfig,
    build_prefix_velocity_policies,
)
from .deform360_replication import load_deform360_replication_protocol
from .deform360_replication_case import (
    build_replication_warp_observation,
    score_replication_warp_prediction,
)
from .deform360_replication_contact import (
    ReplicationOpeningContactModel,
    contact_state_by_robot_axis,
    load_replication_contact_episode,
)
from .deform360_replication_fit import validate_source_warp_candidate_grid
from .deform360_replication_geometry import load_replication_hull_archive
from .deform360_replication_warp import (
    OfficialWarpSparseGraphRunner,
    sparse_graph_strain_summary,
)


PREFIX_KINEMATICS_DIAGNOSTIC_SCHEMA_VERSION = 1
PREFIX_KINEMATICS_DIAGNOSTIC_KIND = (
    "Deform360SourcePrefixKinematicsDiagnostic"
)
PREFIX_KINEMATICS_CONFIG_KIND = "Deform360SourcePrefixKinematicsConfig"
SOURCE_MILESTONE = Path("milestones/deform360-replication-source-backend-v1")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
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
        type(value) in {int, float}
        and type(value) is not bool
        and np.isfinite(value),
        f"{name} must be a finite number",
    )
    return float(value)


def _positive_finite_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    _require(result > 0.0, f"{name} must be positive")
    return result


def _strict_nonnegative_integer(value: Any, *, name: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{name} must be a nonnegative integer",
    )
    return value


@dataclass(frozen=True)
class PrefixKinematicsDiagnosticConfig:
    """Predeclared source-only comparison and gate controls."""

    kinematics: PrefixKinematicsConfig = field(
        default_factory=PrefixKinematicsConfig
    )
    baseline_chamfer_tolerance_m: float = 5e-4
    baseline_strain_tolerance: float = 1e-2
    minimum_relative_improvement: float = 0.05
    minimum_win_fraction: float = 0.60
    minimum_common_episode_count: int = 24

    def __post_init__(self) -> None:
        _require(
            type(self.kinematics) is PrefixKinematicsConfig,
            "kinematics must be PrefixKinematicsConfig",
        )
        for name in (
            "baseline_chamfer_tolerance_m",
            "baseline_strain_tolerance",
            "minimum_relative_improvement",
            "minimum_win_fraction",
        ):
            value = _finite_float(getattr(self, name), name=name)
            _require(value >= 0.0, f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        _require(
            self.minimum_relative_improvement < 1.0,
            "minimum_relative_improvement must be below one",
        )
        _require(
            self.minimum_win_fraction <= 1.0,
            "minimum_win_fraction must be at most one",
        )
        object.__setattr__(
            self,
            "minimum_common_episode_count",
            _strict_nonnegative_integer(
                self.minimum_common_episode_count,
                name="minimum_common_episode_count",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kinematics"] = self.kinematics.as_dict()
        return payload


def prefix_kinematics_config_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a diagnostic lock without its self-reported digest."""

    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def load_prefix_kinematics_diagnostic_lock(
    path: str | Path,
) -> dict[str, Any]:
    """Load the exact source-only diagnostic controls and object set."""

    lock_path = Path(path).resolve()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == 1,
        "prefix-kinematics config schema changed",
    )
    _require(
        payload.get("artifact_kind") == PREFIX_KINEMATICS_CONFIG_KIND,
        "prefix-kinematics config kind changed",
    )
    _require(
        payload.get("config_sha256")
        == prefix_kinematics_config_sha256(payload),
        "prefix-kinematics config checksum mismatch",
    )
    controls = payload.get("config")
    _require(isinstance(controls, Mapping), "prefix-kinematics controls are missing")
    kinematics = controls.get("kinematics")
    _require(isinstance(kinematics, Mapping), "kinematics controls are missing")
    selected = controls.get("selected_object_ids")
    _require(
        isinstance(selected, list)
        and selected
        and all(type(value) is str and value for value in selected)
        and len(selected) == len(set(selected)),
        "selected source object set is invalid",
    )
    _require(
        controls.get("candidate_selection")
        == "quality_constrained_source_oracle_else_finite_oracle",
        "source candidate-selection policy changed",
    )
    _require(
        controls.get("primary_policy") == "graph_harmonic_contact_v1"
        and controls.get("control_policy")
        == "global_contact_translation_v1",
        "prefix-kinematics policy roles changed",
    )
    boundary = payload.get("information_boundary")
    _require(isinstance(boundary, Mapping), "config information boundary is missing")
    _require(
        boundary.get("source_only") is True
        and boundary.get("calibration_outcomes_allowed") is False
        and boundary.get("target_prefix_allowed") is False
        and boundary.get("target_future_allowed") is False,
        "config opens forbidden replication outcomes",
    )
    diagnostic = PrefixKinematicsDiagnosticConfig(
        kinematics=PrefixKinematicsConfig(**dict(kinematics)),
        baseline_chamfer_tolerance_m=controls[
            "baseline_chamfer_tolerance_m"
        ],
        baseline_strain_tolerance=controls["baseline_strain_tolerance"],
        minimum_relative_improvement=controls[
            "minimum_relative_improvement"
        ],
        minimum_win_fraction=controls["minimum_win_fraction"],
        minimum_common_episode_count=controls[
            "minimum_common_episode_count"
        ],
    )
    return {
        "path": lock_path,
        "payload": payload,
        "config": diagnostic,
        "selected_object_ids": tuple(selected),
    }


def verify_source_milestone(repository_root: str | Path) -> dict[str, Any]:
    """Verify every file bound by the frozen source-backend manifest."""

    root = Path(repository_root).resolve()
    milestone = root / SOURCE_MILESTONE
    manifest_path = milestone / "artifact-manifest.json"
    _require(manifest_path.is_file(), "source milestone manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    _require(isinstance(entries, list) and entries, "source manifest has no entries")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        _require(isinstance(entry, Mapping), "source manifest entry is not a mapping")
        source_path = entry.get("source_path")
        _require(
            type(source_path) is str and source_path not in seen,
            "source manifest path is invalid or repeated",
        )
        seen.add(source_path)
        path = root / source_path
        _require(path.is_file(), f"source milestone file is missing: {source_path}")
        _require(
            path.stat().st_size == entry.get("bytes"),
            f"source milestone byte count changed: {source_path}",
        )
        _require(
            _sha256_file(path) == entry.get("sha256"),
            f"source milestone checksum changed: {source_path}",
        )
        _require(
            type(entry.get("id")) is str and entry["id"],
            f"source milestone entry {index} has no id",
        )
    return {
        "manifest_path": str(manifest_path.relative_to(root)),
        "manifest_sha256": _sha256_file(manifest_path),
        "verified_file_count": len(entries),
    }


def select_fixed_source_candidate(
    source_grid: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one candidate using only the already-open source-grid outcome."""

    validate_source_warp_candidate_grid(source_grid)
    limit = _positive_finite_float(
        source_grid["config"]["maximum_p99_relative_edge_strain"],
        name="maximum_p99_relative_edge_strain",
    )
    finite_rows: list[tuple[float, int, Mapping[str, Any]]] = []
    quality_rows: list[tuple[float, int, Mapping[str, Any]]] = []
    for row in source_grid["candidate_scores"]:
        index = _strict_nonnegative_integer(
            row.get("candidate_index"),
            name="candidate_index",
        )
        if row.get("finite") is not True:
            continue
        score = row.get("mean_chamfer_m")
        strain = row.get("p99_relative_edge_strain")
        if (
            type(score) not in {int, float}
            or type(score) is bool
            or not np.isfinite(score)
            or type(strain) not in {int, float}
            or type(strain) is bool
            or not np.isfinite(strain)
        ):
            continue
        candidate = (float(score), index, row)
        finite_rows.append(candidate)
        if float(strain) <= limit:
            quality_rows.append(candidate)
    _require(finite_rows, "source grid has no finite candidate")
    selected_pool = quality_rows if quality_rows else finite_rows
    score, index, row = min(selected_pool, key=lambda value: (value[0], value[1]))
    parameters = row.get("parameters")
    _require(isinstance(parameters, Mapping), "selected candidate has no parameters")
    return {
        "selection_kind": (
            "quality_constrained_source_oracle"
            if quality_rows
            else "finite_unconstrained_source_oracle"
        ),
        "candidate_index": index,
        "parameters": dict(parameters),
        "archived_mean_chamfer_m": score,
        "archived_late_chamfer_m": _finite_float(
            row["late_chamfer_m"],
            name="archived_late_chamfer_m",
        ),
        "archived_p99_relative_edge_strain": _finite_float(
            row["p99_relative_edge_strain"],
            name="archived_p99_relative_edge_strain",
        ),
        "archived_quality_valid": bool(
            np.isfinite(row["p99_relative_edge_strain"])
            and float(row["p99_relative_edge_strain"]) <= limit
        ),
        "maximum_p99_relative_edge_strain": limit,
    }


def summarize_policy(
    episode_records: Sequence[Mapping[str, Any]],
    policy_name: str,
) -> dict[str, Any]:
    """Aggregate one fixed policy against the rerun zero-velocity baseline."""

    _require(
        policy_name in PREFIX_KINEMATICS_POLICIES,
        "unknown prefix-kinematics policy",
    )
    scores: list[float] = []
    zero_scores: list[float] = []
    wins: list[bool] = []
    quality_valid = 0
    zero_quality_valid = 0
    rescues = 0
    regressions = 0
    per_object: dict[str, list[tuple[float, float, bool, bool]]] = {}
    for record in episode_records:
        policies = record.get("policies")
        _require(isinstance(policies, Mapping), "episode policy record is missing")
        baseline = policies.get(PREFIX_KINEMATICS_POLICIES[0])
        candidate = policies.get(policy_name)
        _require(
            isinstance(baseline, Mapping) and isinstance(candidate, Mapping),
            "episode policy result is missing",
        )
        if baseline.get("finite") is not True or candidate.get("finite") is not True:
            continue
        baseline_score = _positive_finite_float(
            baseline["mean_chamfer_m"],
            name="zero mean_chamfer_m",
        )
        candidate_score = _positive_finite_float(
            candidate["mean_chamfer_m"],
            name="candidate mean_chamfer_m",
        )
        baseline_quality = baseline.get("quality_valid") is True
        candidate_quality = candidate.get("quality_valid") is True
        scores.append(candidate_score)
        zero_scores.append(baseline_score)
        wins.append(bool(candidate_quality and candidate_score < baseline_score))
        zero_quality_valid += int(baseline_quality)
        quality_valid += int(candidate_quality)
        rescues += int(not baseline_quality and candidate_quality)
        regressions += int(baseline_quality and not candidate_quality)
        object_id = str(record["object_id"])
        per_object.setdefault(object_id, []).append(
            (
                candidate_score,
                baseline_score,
                candidate_quality,
                candidate_quality and candidate_score < baseline_score,
            )
        )
    if not scores:
        return {
            "policy": policy_name,
            "common_finite_episode_count": 0,
            "mean_chamfer_m": None,
            "zero_mean_chamfer_m": None,
            "relative_improvement_vs_zero": None,
            "win_fraction_vs_zero": None,
            "quality_valid_episode_count": 0,
            "zero_quality_valid_episode_count": 0,
            "quality_rescue_count": 0,
            "quality_regression_count": 0,
            "per_object": {},
        }
    mean_score = float(np.mean(scores))
    mean_zero = float(np.mean(zero_scores))
    object_summary = {}
    for object_id, rows in sorted(per_object.items()):
        object_scores = [row[0] for row in rows]
        object_zero = [row[1] for row in rows]
        object_summary[object_id] = {
            "episode_count": len(rows),
            "mean_chamfer_m": float(np.mean(object_scores)),
            "zero_mean_chamfer_m": float(np.mean(object_zero)),
            "relative_improvement_vs_zero": (
                float(np.mean(object_zero)) - float(np.mean(object_scores))
            )
            / float(np.mean(object_zero)),
            "quality_valid_episode_count": int(sum(row[2] for row in rows)),
            "win_fraction_vs_zero": float(np.mean([row[3] for row in rows])),
        }
    return {
        "policy": policy_name,
        "common_finite_episode_count": len(scores),
        "mean_chamfer_m": mean_score,
        "zero_mean_chamfer_m": mean_zero,
        "relative_improvement_vs_zero": (mean_zero - mean_score) / mean_zero,
        "win_fraction_vs_zero": float(np.mean(wins)),
        "quality_valid_episode_count": quality_valid,
        "zero_quality_valid_episode_count": zero_quality_valid,
        "quality_rescue_count": rescues,
        "quality_regression_count": regressions,
        "per_object": object_summary,
    }


def build_source_decision(
    episode_records: Sequence[Mapping[str, Any]],
    *,
    config: PrefixKinematicsDiagnosticConfig,
) -> dict[str, Any]:
    """Apply the predeclared source-only gate to graph-harmonic initialization."""

    reproduction = [
        bool(record.get("zero_baseline_reproduction", {}).get("passed"))
        for record in episode_records
    ]
    summaries = {
        policy: summarize_policy(episode_records, policy)
        for policy in PREFIX_KINEMATICS_POLICIES
    }
    primary = summaries["graph_harmonic_contact_v1"]
    improvement = primary["relative_improvement_vs_zero"]
    win_fraction = primary["win_fraction_vs_zero"]
    passed = bool(
        reproduction
        and all(reproduction)
        and primary["common_finite_episode_count"]
        >= config.minimum_common_episode_count
        and improvement is not None
        and improvement >= config.minimum_relative_improvement
        and win_fraction is not None
        and win_fraction >= config.minimum_win_fraction
        and primary["quality_valid_episode_count"]
        >= primary["zero_quality_valid_episode_count"]
    )
    return {
        "primary_policy": "graph_harmonic_contact_v1",
        "baseline_reproduction_passed": bool(reproduction and all(reproduction)),
        "baseline_reproduction_episode_count": len(reproduction),
        "minimum_common_episode_count": config.minimum_common_episode_count,
        "minimum_relative_improvement": config.minimum_relative_improvement,
        "minimum_win_fraction": config.minimum_win_fraction,
        "require_non_decreasing_quality_valid_count": True,
        "policy_summaries": summaries,
        "passed": passed,
        "interpretation": (
            "causal prefix kinematics is a source-supported backend repair candidate"
            if passed
            else "causal prefix kinematics does not pass the source-only repair gate"
        ),
        "target_prefix_access_permitted": False,
        "target_future_access_permitted": False,
    }


def _verify_contact_associations(
    archived: Sequence[Mapping[str, Any]],
    reconstructed: Sequence[Mapping[str, Any]],
) -> None:
    _require(
        len(archived) == len(reconstructed),
        "contact association count changed",
    )
    for expected, observed in zip(archived, reconstructed, strict=True):
        for name in (
            "robot_axis",
            "contact_node_index",
            "selected_taxel_indices",
        ):
            _require(
                expected[name] == observed[name],
                f"contact association {name} changed",
            )
        _require(
            np.allclose(
                expected["contact_offset_m"],
                observed["contact_offset_m"],
                rtol=0.0,
                atol=1e-12,
            ),
            "contact association offset changed",
        )


def _clear_optional_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - integration environment
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _policy_result(
    observation,
    official_phystwin_repo: Path,
    simulation_config: WarpRopeFeasibilityConfig,
    candidate: WarpRopeCandidate,
    initial_velocity_m_s: np.ndarray,
    *,
    device: str,
) -> dict[str, Any]:
    case = replace(
        observation.case,
        initial_velocities_m_s=initial_velocity_m_s,
    )
    runner = OfficialWarpSparseGraphRunner(
        official_phystwin_repo,
        case,
        simulation_config,
        device=device,
    )
    prediction = runner.rollout(candidate)
    metrics = score_replication_warp_prediction(observation, prediction)
    strain = sparse_graph_strain_summary(observation.case.graph, prediction)
    mean = float(metrics["mean_m"])
    late = float(metrics["late_mean_m"])
    p99 = float(strain["p99"])
    maximum = float(strain["maximum"])
    del runner, prediction
    _clear_optional_cuda_cache()
    return {
        "finite": bool(
            np.isfinite(mean)
            and np.isfinite(late)
            and np.isfinite(p99)
            and np.isfinite(maximum)
        ),
        "mean_chamfer_m": mean if np.isfinite(mean) else None,
        "late_chamfer_m": late if np.isfinite(late) else None,
        "p99_relative_edge_strain": p99 if np.isfinite(p99) else None,
        "maximum_relative_edge_strain": (
            maximum if np.isfinite(maximum) else None
        ),
        "quality_valid": bool(
            np.isfinite(mean)
            and np.isfinite(p99)
            and p99 <= simulation_config.maximum_p99_relative_edge_strain
        ),
    }


def _episode_record(
    *,
    repository_root: Path,
    data_root: Path,
    source_grid_path: Path,
    cohort: Mapping[str, Any],
    official_phystwin_repo: Path,
    device: str,
    config: PrefixKinematicsDiagnosticConfig,
) -> dict[str, Any]:
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
        "diagnostic was given a non-source episode",
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
    hulls = tuple(
        hull for hull, keep in zip(hulls, available, strict=True) if keep
    )
    _require(
        total_frame_count == grid["reference_geometry_total_frame_count"]
        and len(hulls) == grid["reference_geometry_available_frame_count"],
        "source hull availability changed",
    )
    _require(
        frames.astype(int).tolist() == grid["raw_hull_frame_indices"],
        "source hull frame indices changed",
    )
    observation = build_replication_warp_observation(
        episode_dir,
        episode_id,
        str(grid["stratum"]),
        frames,
        hulls,
        schedule,
    )
    _verify_contact_associations(
        grid["contact_associations"],
        observation.contact_associations,
    )
    velocity_policies, kinematic_diagnostics = build_prefix_velocity_policies(
        observation.case.graph,
        episode_dir,
        prefix_endpoint_frame=observation.prefix_endpoint_frame,
        contact_associations=observation.contact_associations,
        full_contact_active=schedule,
        contact_node_indices=observation.case.contact_node_indices,
        dt_seconds=observation.case.dt_seconds,
        config=config.kinematics,
    )
    simulation_config = WarpRopeFeasibilityConfig(**grid["config"])
    candidate = WarpRopeCandidate(**selected["parameters"])
    policies = {
        name: _policy_result(
            observation,
            official_phystwin_repo,
            simulation_config,
            candidate,
            velocity,
            device=device,
        )
        for name, velocity in velocity_policies.items()
    }
    zero = policies["zero_v1"]
    mean_delta = (
        abs(float(zero["mean_chamfer_m"]) - selected["archived_mean_chamfer_m"])
        if zero["mean_chamfer_m"] is not None
        else None
    )
    strain_delta = (
        abs(
            float(zero["p99_relative_edge_strain"])
            - selected["archived_p99_relative_edge_strain"]
        )
        if zero["p99_relative_edge_strain"] is not None
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
        "kinematics": kinematic_diagnostics,
        "policies": policies,
        "zero_baseline_reproduction": {
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
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
            "velocity_uses_frames_no_later_than_prefix_endpoint": True,
        },
    }


def run_source_prefix_kinematics_diagnostic(
    repository_root: str | Path,
    protocol_path: str | Path,
    data_root: str | Path,
    official_phystwin_repo: str | Path,
    output_path: str | Path,
    *,
    lock_path: str | Path | None = None,
    object_ids: Sequence[str] | None = None,
    device: str = "cuda:0",
    config: PrefixKinematicsDiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Run the sealed zero/global/graph initial-velocity source comparison."""

    lock = (
        load_prefix_kinematics_diagnostic_lock(lock_path)
        if lock_path is not None
        else None
    )
    _require(
        lock is None or (object_ids is None and config is None),
        "a locked run cannot override its objects or controls",
    )
    cfg = (
        lock["config"]
        if lock is not None
        else config or PrefixKinematicsDiagnosticConfig()
    )
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
    cohorts = {
        str(record["object_id"]): record
        for record in protocol["config"]["cohort"]
    }
    grid_root = root / SOURCE_MILESTONE / "artifacts" / "source-grids"
    _require(grid_root.is_dir(), "source-grid milestone directory is missing")
    selected_objects = (
        tuple(lock["selected_object_ids"])
        if lock is not None
        else (
            tuple(object_ids)
            if object_ids is not None
            else tuple(
                sorted(
                    path.name
                    for path in grid_root.iterdir()
                    if path.is_dir()
                    and any(path.glob("source_episode_*_grid.json"))
                )
            )
        )
    )
    _require(selected_objects, "diagnostic object set is empty")
    _require(
        len(selected_objects) == len(set(selected_objects))
        and all(type(value) is str and value in cohorts for value in selected_objects),
        "diagnostic object set is invalid",
    )
    if lock is not None:
        lock_payload = lock["payload"]
        _require(
            lock_payload["protocol_config_sha256"]
            == protocol["config_sha256"],
            "locked replication protocol identity changed",
        )
        _require(
            lock_payload["source_milestone_manifest_sha256"]
            == milestone_verification["manifest_sha256"],
            "locked source milestone identity changed",
        )
        decision_path = (
            root
            / SOURCE_MILESTONE
            / "artifacts"
            / "source_backend_decision.json"
        )
        source_decision = json.loads(
            decision_path.read_text(encoding="utf-8")
        )
        _require(
            lock_payload["source_backend_decision_result_sha256"]
            == source_decision["result_sha256"],
            "locked source-backend decision identity changed",
        )
    records = []
    for object_id in selected_objects:
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
                    config=cfg,
                )
            )
    decision = build_source_decision(records, config=cfg)
    payload = {
        "schema_version": PREFIX_KINEMATICS_DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_kind": PREFIX_KINEMATICS_DIAGNOSTIC_KIND,
        "config": cfg.as_dict(),
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
        "diagnostic_lock": (
            {
                "path": (
                    str(lock["path"].relative_to(root))
                    if lock is not None and lock["path"].is_relative_to(root)
                    else str(lock["path"])
                ),
                "file_sha256": _sha256_file(lock["path"]),
                "config_sha256": lock["payload"]["config_sha256"],
            }
            if lock is not None
            else None
        ),
        "selected_object_ids": list(selected_objects),
        "objects_without_complete_source_grids": sorted(
            set(cohorts) - set(selected_objects)
        ),
        "episode_records": records,
        "decision": decision,
        "information_boundary": {
            "source_candidate_outcomes_read": True,
            "source_future_geometry_read_for_scoring": True,
            "source_tactile_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
            "registered_replication_result_changed": False,
        },
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    atomic_write_json(output_path, payload)
    return payload


def validate_source_prefix_kinematics_diagnostic(
    payload: Mapping[str, Any],
) -> None:
    """Validate the immutable diagnostic result and target-closed boundary."""

    _require(
        payload.get("schema_version")
        == PREFIX_KINEMATICS_DIAGNOSTIC_SCHEMA_VERSION,
        "prefix-kinematics diagnostic schema changed",
    )
    _require(
        payload.get("artifact_kind") == PREFIX_KINEMATICS_DIAGNOSTIC_KIND,
        "prefix-kinematics diagnostic kind changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "prefix-kinematics diagnostic checksum mismatch",
    )
    boundary = payload.get("information_boundary")
    _require(isinstance(boundary, Mapping), "diagnostic boundary is missing")
    _require(
        boundary.get("calibration_outcomes_read") is False
        and boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False
        and boundary.get("registered_replication_result_changed") is False,
        "diagnostic crossed its information or claim boundary",
    )
    records = payload.get("episode_records")
    _require(isinstance(records, list) and records, "diagnostic has no episodes")
    for record in records:
        _require(
            record["information_boundary"]["source_episode_only"] is True
            and record["information_boundary"]["target_prefix_read"] is False
            and record["information_boundary"]["target_future_read"] is False,
            "episode crossed the source-only boundary",
        )
        _require(
            tuple(record["policies"]) == PREFIX_KINEMATICS_POLICIES,
            "diagnostic policy set or ordering changed",
        )


__all__ = [
    "PREFIX_KINEMATICS_DIAGNOSTIC_KIND",
    "PREFIX_KINEMATICS_DIAGNOSTIC_SCHEMA_VERSION",
    "PREFIX_KINEMATICS_CONFIG_KIND",
    "PrefixKinematicsDiagnosticConfig",
    "load_prefix_kinematics_diagnostic_lock",
    "prefix_kinematics_config_sha256",
    "build_source_decision",
    "run_source_prefix_kinematics_diagnostic",
    "select_fixed_source_candidate",
    "summarize_policy",
    "validate_source_prefix_kinematics_diagnostic",
    "verify_source_milestone",
]
