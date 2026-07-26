"""Consume the public Bayesian-PhysTwin provider API for Causal4D beliefs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from bayesian_phystwin.causal4d_provider_v1 import (
    BPTBeliefExportConfigV1,
    PhysicalBeliefV1,
    ProviderManifestV1,
    build_physical_belief_from_replays,
    released_self_collision_for_case,
    replay_official_phystwin_particles,
    target_validity,
)
from causal4d.contracts import CausalContext, TwinBelief

if TYPE_CHECKING:
    from causal4d.phystwin_backend import OfficialPhysTwinBackend

BPT_PROVIDER_REVISION = "aed1de60a0fca195bdd227fe598cb4eb65f113b9"
BPTBeliefExportConfig = BPTBeliefExportConfigV1


def lift_isotropic_discrepancy_variance(
    tracked_variance_m2: np.ndarray,
    state_count: int,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Propagate independent tracked variances through a fixed kNN readout."""

    tracked = np.asarray(tracked_variance_m2, dtype=float)
    neighbor_indices = np.asarray(indices, dtype=np.int64)
    neighbor_weights = np.asarray(weights, dtype=float)
    if (
        tracked.ndim != 1
        or not np.all(np.isfinite(tracked))
        or np.any(tracked < 0.0)
    ):
        raise ValueError(
            "tracked_variance_m2 must be a finite nonnegative vector"
        )
    if state_count < len(tracked):
        raise ValueError("state_count cannot be smaller than the tracked state")
    extra_count = state_count - len(tracked)
    if (
        neighbor_indices.shape != neighbor_weights.shape
        or neighbor_indices.shape[0] != extra_count
    ):
        raise ValueError("lift map must identify every untracked state node")
    if np.any(neighbor_indices < 0) or np.any(
        neighbor_indices >= len(tracked)
    ):
        raise ValueError("lift map references an unavailable tracked node")
    if extra_count and not np.allclose(
        np.sum(neighbor_weights, axis=1), 1.0
    ):
        raise ValueError("lift weights must sum to one")
    scalar = np.empty(state_count, dtype=float)
    scalar[: len(tracked)] = tracked
    if extra_count:
        scalar[len(tracked) :] = np.sum(
            np.square(neighbor_weights) * tracked[neighbor_indices],
            axis=1,
        )
    return np.repeat(scalar[:, None], 3, axis=1)


def _provider_manifest() -> ProviderManifestV1:
    return ProviderManifestV1(
        provider_revision=BPT_PROVIDER_REVISION,
        metadata={
            "consumer": "causal4d",
            "contract": "causal4d_provider_v1",
        },
    )


def _maximum_pairwise_endpoint_rmse(positions: np.ndarray) -> float:
    values = np.asarray(positions, dtype=float)
    maximum = 0.0
    for first in range(len(values)):
        for second in range(first + 1, len(values)):
            maximum = max(
                maximum,
                float(
                    np.sqrt(
                        np.mean(
                            np.square(values[first] - values[second])
                        )
                    )
                ),
            )
    return maximum


def _to_twin_belief(
    context: CausalContext,
    physical: PhysicalBeliefV1,
) -> TwinBelief:
    if physical.endpoint_frame != context.o_minus.frame_stop - 1:
        raise ValueError(
            "provider belief endpoint does not match Causal4D O-"
        )
    metadata = dict(physical.metadata)
    metadata.update(
        {
            "bpt_provider_revision": BPT_PROVIDER_REVISION,
            "bpt_provider_manifest_id": physical.provider_manifest_id,
            "bpt_physical_belief_id": physical.artifact_id,
            "bpt_provider_api": (
                "bayesian_phystwin.causal4d_provider_v1"
            ),
        }
    )
    return TwinBelief(
        context=context,
        endpoint_frame=physical.endpoint_frame,
        particle_ids=physical.particle_ids,
        theta_names=physical.theta_names,
        endpoint_position_m=physical.endpoint_position_m,
        endpoint_velocity_mps=physical.endpoint_velocity_mps,
        theta=physical.theta,
        discrepancy_mean_m=physical.discrepancy_mean_m,
        discrepancy_variance_m2=physical.discrepancy_variance_m2,
        weights=physical.weights,
        metadata=metadata,
    )


def build_twin_belief_from_replays(
    *,
    context: CausalContext,
    replay_positions_m: np.ndarray,
    replay_velocities_mps: np.ndarray,
    observed_positions_m: np.ndarray,
    observed_valid: np.ndarray,
    theta: np.ndarray,
    theta_names: tuple[str, ...],
    weights: np.ndarray,
    particle_ids: tuple[str, ...] | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: BPTBeliefExportConfig | None = None,
) -> TwinBelief:
    """Build a Causal4D belief through the public BPT provider contract."""

    positions = np.asarray(replay_positions_m, dtype=float)
    diagnostics = dict(metadata or {})
    diagnostics.update(
        {
            "causal_context": context.as_dict(),
            "maximum_pairwise_endpoint_rmse_m": (
                _maximum_pairwise_endpoint_rmse(
                    positions[:, context.o_minus.frame_stop - 1]
                )
            ),
        }
    )
    manifest = _provider_manifest()
    physical = build_physical_belief_from_replays(
        provider_manifest_id=manifest.manifest_id,
        causal_frame_stop=context.o_minus.frame_stop,
        replay_positions_m=positions,
        replay_velocities_mps=replay_velocities_mps,
        observed_positions_m=observed_positions_m,
        observed_valid=observed_valid,
        theta=theta,
        theta_names=theta_names,
        weights=weights,
        particle_ids=particle_ids,
        metadata=diagnostics,
        config=config,
    )
    return _to_twin_belief(context, physical)


def export_official_phystwin_twin_belief(
    backend: OfficialPhysTwinBackend,
    *,
    context: CausalContext,
    config: BPTBeliefExportConfig | None = None,
) -> TwinBelief:
    """Replay selected particles through the public provider execution API."""

    if context.case_id != backend.case_name:
        raise ValueError(
            "causal context case does not match the PhysTwin backend"
        )
    if (
        context.o_minus.frame_start != 0
        or context.o_minus.frame_stop != backend.train_end_frame
    ):
        raise ValueError(
            "causal context O- does not match the backend training split"
        )
    self_collision = (
        released_self_collision_for_case(backend.case_name)
        if backend.config.self_collision is None
        else bool(backend.config.self_collision)
    )
    replay_positions, replay_velocities = (
        replay_official_phystwin_particles(
            official_repo=backend.official_repo,
            data=backend.data,
            optimal=backend.optimal,
            checkpoint_path=backend.checkpoint_path,
            graph=backend.graph,
            log_scales=backend.particles.log_scales,
            original_count=backend.original_count,
            surface_point_count=len(backend.surface_points),
            frame_count=backend.train_end_frame,
            dt=backend.config.dt,
            num_substeps=backend.config.num_substeps,
            self_collision=self_collision,
            deterministic_spring_forces=(
                backend.config.deterministic_spring_forces
            ),
            device=backend.config.device,
        )
    )
    valid = target_validity(backend.visible, backend.motion_valid)
    particle_ids = tuple(
        "grid_" + "_".join(map(str, grid_index))
        for grid_index in backend.particles.grid_indices
    )
    return build_twin_belief_from_replays(
        context=context,
        replay_positions_m=replay_positions,
        replay_velocities_mps=replay_velocities,
        observed_positions_m=backend.object_points,
        observed_valid=valid,
        theta=backend.particles.log_scales,
        theta_names=(
            "object_spring_log_scale",
            "controller_spring_log_scale",
        ),
        weights=backend.particles.weights,
        particle_ids=particle_ids,
        metadata={
            "profile_path": str(backend.profile_path.resolve()),
            "profile_weight_key": backend.particles.source_weight_key,
            "profile_support_method": (
                backend.particles.selection_method
            ),
            "profile_retained_probability_mass": (
                backend.particles.retained_probability_mass
            ),
            "profile_represented_probability_mass": (
                backend.particles.represented_probability_mass
            ),
            "official_backend": backend.default_manifest(),
        },
        config=config,
    )


__all__ = [
    "BPTBeliefExportConfig",
    "BPT_PROVIDER_REVISION",
    "build_twin_belief_from_replays",
    "export_official_phystwin_twin_belief",
    "lift_isotropic_discrepancy_variance",
]
