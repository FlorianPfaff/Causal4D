"""Per-episode execution for the Deform360 contact/support diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_contact_support_contract import (
    CONTACT_SUPPORT_POLICIES,
    ContactSupportDiagnosticConfig,
    _finite_float,
    _require,
    _require_mapping,
)
from .deform360_phystwin_feasibility import (
    WarpRopeCandidate,
    WarpRopeFeasibilityConfig,
)
from .deform360_prefix_kinematics_diagnostic import select_fixed_source_candidate
from .deform360_replication_case import (
    ReplicationWarpObservation,
    build_replication_warp_observation,
    score_replication_warp_prediction,
)
from .deform360_replication_contact import (
    ReplicationOpeningContactModel,
    contact_state_by_robot_axis,
    load_replication_contact_episode,
    visual_contact_schedule,
)
from .deform360_replication_fit import validate_source_warp_candidate_grid
from .deform360_replication_geometry import load_replication_hull_archive
from .deform360_replication_warp import (
    OfficialWarpSparseGraphRunner,
    sparse_graph_strain_summary,
)


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


def _run_policy(
    observation: ReplicationWarpObservation,
    official_phystwin_repo: Path,
    simulation_config: WarpRopeFeasibilityConfig,
    candidate: WarpRopeCandidate,
    *,
    device: str,
) -> dict[str, Any]:
    runner = OfficialWarpSparseGraphRunner(
        official_phystwin_repo,
        observation.case,
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
        "maximum_relative_edge_strain": maximum if np.isfinite(maximum) else None,
        "quality_valid": bool(
            np.isfinite(mean)
            and np.isfinite(p99)
            and p99 <= simulation_config.maximum_p99_relative_edge_strain
        ),
    }


def _association_summary(observation: ReplicationWarpObservation) -> dict[str, Any]:
    return {
        "contact_node_indices": list(observation.case.contact_node_indices),
        "selected_taxel_counts": [
            len(record["selected_taxel_indices"])
            for record in observation.contact_associations
        ],
        "patch_to_node_distance_m": [
            float(record["patch_to_node_distance_m"])
            for record in observation.contact_associations
        ],
        "nearest_surface_distance_m": [
            float(record["nearest_surface_distance_m"])
            for record in observation.contact_associations
        ],
    }


def _load_source_episode(
    *,
    data_root: Path,
    source_grid_path: Path,
    cohort: Mapping[str, Any],
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
    tactile_schedule = contact_state_by_robot_axis(
        episode,
        contact_model.tactile_group_to_robot_axis,
    )
    opening_schedule = visual_contact_schedule(episode, contact_model)
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
    return {
        "grid": grid,
        "selected": selected,
        "object_id": object_id,
        "episode_id": episode_id,
        "episode_dir": episode_dir,
        "stratum": str(grid["stratum"]),
        "frames": frames,
        "hulls": hulls,
        "tactile_schedule": tactile_schedule,
        "opening_schedule": opening_schedule,
        "simulation_config": WarpRopeFeasibilityConfig(**grid["config"]),
    }


def build_contact_support_episode_record(
    *,
    repository_root: Path,
    data_root: Path,
    source_grid_path: Path,
    cohort: Mapping[str, Any],
    official_phystwin_repo: Path,
    device: str,
    config: ContactSupportDiagnosticConfig,
) -> dict[str, Any]:
    """Run every locked mechanism for one already-opened source episode."""

    loaded = _load_source_episode(
        data_root=data_root,
        source_grid_path=source_grid_path,
        cohort=cohort,
    )
    grid = loaded["grid"]
    tactile_schedule = np.asarray(loaded["tactile_schedule"], dtype=bool)
    opening_schedule = np.asarray(loaded["opening_schedule"], dtype=bool)
    zero_schedule = np.zeros_like(tactile_schedule)
    common_arguments = (
        loaded["episode_dir"],
        loaded["episode_id"],
        loaded["stratum"],
        loaded["frames"],
        loaded["hulls"],
    )
    registered = build_replication_warp_observation(
        *common_arguments,
        tactile_schedule,
        selected_taxel_count=config.registered_taxel_count,
    )
    _verify_contact_associations(
        grid["contact_associations"],
        registered.contact_associations,
    )
    narrow = build_replication_warp_observation(
        *common_arguments,
        tactile_schedule,
        selected_taxel_count=config.narrow_taxel_count,
    )
    wide = build_replication_warp_observation(
        *common_arguments,
        tactile_schedule,
        selected_taxel_count=config.wide_taxel_count,
    )
    opening = build_replication_warp_observation(
        *common_arguments,
        opening_schedule,
        selected_taxel_count=config.registered_taxel_count,
    )
    disabled = build_replication_warp_observation(
        *common_arguments,
        zero_schedule,
        selected_taxel_count=config.registered_taxel_count,
    )
    _verify_contact_associations(
        registered.contact_associations,
        opening.contact_associations,
    )
    _verify_contact_associations(
        registered.contact_associations,
        disabled.contact_associations,
    )
    simulation_config = loaded["simulation_config"]
    candidate = WarpRopeCandidate(**loaded["selected"]["parameters"])
    policy_inputs = {
        "registered_v1": (registered, simulation_config),
        "support_touching_v1": (
            registered,
            replace(
                simulation_config,
                initial_ground_clearance_m=config.touching_clearance_m,
            ),
        ),
        "support_lifted_5mm_v1": (
            registered,
            replace(
                simulation_config,
                initial_ground_clearance_m=config.lifted_clearance_m,
            ),
        ),
        "contact_patch_4_v1": (narrow, simulation_config),
        "contact_patch_12_v1": (wide, simulation_config),
        "opening_contact_schedule_v1": (opening, simulation_config),
        "contact_disabled_v1": (disabled, simulation_config),
    }
    _require(
        tuple(policy_inputs) == CONTACT_SUPPORT_POLICIES,
        "contact/support execution policy order changed",
    )
    policies = {
        name: _run_policy(
            observation,
            official_phystwin_repo,
            policy_config,
            candidate,
            device=device,
        )
        for name, (observation, policy_config) in policy_inputs.items()
    }
    baseline = policies["registered_v1"]
    selected = loaded["selected"]
    mean_delta = (
        abs(float(baseline["mean_chamfer_m"]) - selected["archived_mean_chamfer_m"])
        if baseline["mean_chamfer_m"] is not None
        else None
    )
    strain_delta = (
        abs(
            float(baseline["p99_relative_edge_strain"])
            - selected["archived_p99_relative_edge_strain"]
        )
        if baseline["p99_relative_edge_strain"] is not None
        else None
    )
    reproduction_passed = bool(
        mean_delta is not None
        and strain_delta is not None
        and mean_delta <= config.baseline_chamfer_tolerance_m
        and strain_delta <= config.baseline_strain_tolerance
    )
    return {
        "object_id": loaded["object_id"],
        "stratum": loaded["stratum"],
        "episode_id": loaded["episode_id"],
        "source_grid_path": str(source_grid_path.relative_to(repository_root)),
        "source_grid_result_sha256": grid["result_sha256"],
        "selected_candidate": selected,
        "mechanism_diagnostics": {
            "registered_initial_ground_clearance_m": (
                simulation_config.initial_ground_clearance_m
            ),
            "touching_initial_ground_clearance_m": config.touching_clearance_m,
            "lifted_initial_ground_clearance_m": config.lifted_clearance_m,
            "tactile_active_fraction": _finite_float(
                float(np.mean(tactile_schedule)),
                name="tactile_active_fraction",
            ),
            "opening_active_fraction": _finite_float(
                float(np.mean(opening_schedule)),
                name="opening_active_fraction",
            ),
            "opening_vs_tactile_disagreement_fraction": _finite_float(
                float(np.mean(opening_schedule != tactile_schedule)),
                name="opening_vs_tactile_disagreement_fraction",
            ),
            "registered_association": _association_summary(registered),
            "narrow_association": _association_summary(narrow),
            "wide_association": _association_summary(wide),
        },
        "policies": policies,
        "registered_baseline_reproduction": {
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
            "source_future_geometry_read_for_scoring": True,
            "source_tactile_read": True,
            "source_robot_openings_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_read": False,
            "method_selection_permitted": False,
        },
    }


__all__ = ["build_contact_support_episode_record"]
