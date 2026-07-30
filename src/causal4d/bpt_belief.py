"""Export full Bayesian-PhysTwin endpoint particles for Causal4D."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bayesian_phystwin.causal4d_belief_provider_v1 import (
    DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1,
    FixedBayesianAnchorConfigV1,
    infer_fixed_bayesian_anchor_endpoint,
)
from bayesian_phystwin.causal4d_provider_v1 import released_self_collision_for_case
from bayesian_phystwin.causal4d_provider_v2 import (
    InitialReplayRequestV1,
    PhysTwinReplayProvider,
    build_lift_map,
    create_official_replay_provider,
    lift_residual,
    target_validity,
)
from causal4d.belief_provider_contract import (
    require_bayesian_phystwin_belief_provider,
)
from causal4d.contracts import CausalContext, TwinBelief, array_sha256
from causal4d.replay_provider_contract import (
    stable_replay_identifier,
    validate_replay_trajectory,
)

if TYPE_CHECKING:
    from causal4d.phystwin_backend import OfficialPhysTwinBackend


@dataclass(frozen=True)
class BPTBeliefExportConfig:
    """Fixed, label-free settings for a full endpoint belief export."""

    process_std_m: float = DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1.process_std_m
    observation_std_m: float = DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1.observation_std_m
    initial_std_m: float = DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1.initial_std_m
    inlier_prior: float = DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1.inlier_prior
    outlier_variance_multiplier: float = (
        DEFAULT_FIXED_BAYESIAN_ANCHOR_CONFIG_V1.outlier_variance_multiplier
    )
    interpolation_neighbors: int = 4
    maximum_discrepancy_m: float = 0.01

    def __post_init__(self) -> None:
        anchor = FixedBayesianAnchorConfigV1(
            process_std_m=self.process_std_m,
            observation_std_m=self.observation_std_m,
            initial_std_m=self.initial_std_m,
            inlier_prior=self.inlier_prior,
            outlier_variance_multiplier=self.outlier_variance_multiplier,
        )
        if (
            isinstance(self.interpolation_neighbors, bool)
            or not isinstance(self.interpolation_neighbors, (int, np.integer))
            or self.interpolation_neighbors < 1
        ):
            raise ValueError("interpolation_neighbors must be a positive integer")
        maximum_discrepancy = float(self.maximum_discrepancy_m)
        if not np.isfinite(maximum_discrepancy) or maximum_discrepancy <= 0.0:
            raise ValueError("maximum_discrepancy_m must be finite and positive")
        object.__setattr__(self, "process_std_m", anchor.process_std_m)
        object.__setattr__(self, "observation_std_m", anchor.observation_std_m)
        object.__setattr__(self, "initial_std_m", anchor.initial_std_m)
        object.__setattr__(self, "inlier_prior", anchor.inlier_prior)
        object.__setattr__(
            self,
            "outlier_variance_multiplier",
            anchor.outlier_variance_multiplier,
        )
        object.__setattr__(
            self, "interpolation_neighbors", int(self.interpolation_neighbors)
        )
        object.__setattr__(self, "maximum_discrepancy_m", maximum_discrepancy)

    def fixed_anchor_config(self) -> FixedBayesianAnchorConfigV1:
        """Return the immutable configuration owned by Bayesian-PhysTwin."""

        return FixedBayesianAnchorConfigV1(
            process_std_m=self.process_std_m,
            observation_std_m=self.observation_std_m,
            initial_std_m=self.initial_std_m,
            inlier_prior=self.inlier_prior,
            outlier_variance_multiplier=self.outlier_variance_multiplier,
        )


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
    if tracked.ndim != 1 or not np.all(np.isfinite(tracked)) or np.any(tracked < 0.0):
        raise ValueError("tracked_variance_m2 must be a finite nonnegative vector")
    if state_count < len(tracked):
        raise ValueError("state_count cannot be smaller than the tracked state")
    extra_count = state_count - len(tracked)
    if (
        neighbor_indices.shape != neighbor_weights.shape
        or neighbor_indices.shape[0] != extra_count
    ):
        raise ValueError("lift map must identify every untracked state node")
    if np.any(neighbor_indices < 0) or np.any(neighbor_indices >= len(tracked)):
        raise ValueError("lift map references an unavailable tracked node")
    if extra_count and not np.allclose(np.sum(neighbor_weights, axis=1), 1.0):
        raise ValueError("lift weights must sum to one")
    scalar = np.empty(state_count, dtype=float)
    scalar[: len(tracked)] = tracked
    if extra_count:
        scalar[len(tracked) :] = np.sum(
            np.square(neighbor_weights) * tracked[neighbor_indices],
            axis=1,
        )
    return np.repeat(scalar[:, None], 3, axis=1)


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
    metadata: dict[str, Any] | None = None,
    config: BPTBeliefExportConfig | None = None,
) -> TwinBelief:
    """Build a belief using only the declared pre-intervention prefix."""

    belief_provider_manifest = require_bayesian_phystwin_belief_provider()
    settings = config or BPTBeliefExportConfig()
    anchor_config = settings.fixed_anchor_config()
    positions = np.asarray(replay_positions_m, dtype=float)
    velocities = np.asarray(replay_velocities_mps, dtype=float)
    observed = np.asarray(observed_positions_m, dtype=float)
    valid = np.asarray(observed_valid, dtype=bool)
    particle_values = np.asarray(theta, dtype=float)
    particle_weights = np.asarray(weights, dtype=float)
    train_end = context.o_minus.frame_stop
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("replay_positions_m must have shape (P, T, N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("replay velocities must match replay positions")
    particle_count, frame_count, state_count, _ = positions.shape
    if frame_count < train_end:
        raise ValueError("replays do not cover O-")
    if observed.ndim != 3 or observed.shape[2] != 3 or len(observed) < train_end:
        raise ValueError("observed_positions_m must cover O- with shape (T, N, 3)")
    tracked_count = observed.shape[1]
    if tracked_count > state_count or valid.shape != observed.shape[:2]:
        raise ValueError("observed validity or tracked state size is inconsistent")
    if particle_values.shape != (particle_count, len(theta_names)):
        raise ValueError("theta does not identify every replay particle")
    if particle_weights.shape != (particle_count,):
        raise ValueError("weights do not identify every replay particle")
    if not 1 <= settings.interpolation_neighbors <= tracked_count:
        raise ValueError("interpolation_neighbors exceeds the tracked point count")

    # Material associations are fixed from the common initial graph geometry.
    lift_indices, lift_weights = build_lift_map(
        positions[0, 0],
        tracked_count,
        settings.interpolation_neighbors,
    )
    discrepancy_means = np.empty((particle_count, state_count, 3), dtype=float)
    discrepancy_variances = np.empty_like(discrepancy_means)
    update_counts: list[int] = []
    final_inlier_probabilities: list[float] = []
    for particle_index in range(particle_count):
        residual = (
            observed[:train_end] - positions[particle_index, :train_end, :tracked_count]
        )
        posterior = infer_fixed_bayesian_anchor_endpoint(
            residual,
            valid[:train_end],
            end_frame=train_end,
            config=anchor_config,
        )
        discrepancy_means[particle_index] = lift_residual(
            posterior.mean_m[None],
            state_count,
            lift_indices,
            lift_weights,
            maximum_norm=settings.maximum_discrepancy_m,
        )[0]
        discrepancy_variances[particle_index] = lift_isotropic_discrepancy_variance(
            posterior.variance_m2,
            state_count,
            lift_indices,
            lift_weights,
        )
        update_counts.append(int(np.sum(posterior.update_count)))
        supported = posterior.update_count > 0
        final_inlier_probabilities.append(
            float(np.mean(posterior.final_nominal_probability[supported]))
            if np.any(supported)
            else 0.0
        )

    endpoint = train_end - 1
    endpoint_positions = positions[:, endpoint].copy()
    endpoint_velocities = velocities[:, endpoint].copy()
    pairwise_rmse = []
    for first in range(particle_count):
        for second in range(first + 1, particle_count):
            pairwise_rmse.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                endpoint_positions[first] - endpoint_positions[second]
                            )
                        )
                    )
                )
            )
    diagnostics = {
        "causal_fit_window": [context.o_minus.frame_start, train_end],
        "future_frames_read_by_estimator": 0,
        "particle_state_source": "official PhysTwin replay through O-",
        "discrepancy_role": "separate readout/process discrepancy; not injected into state",
        "discrepancy_filter": asdict(settings),
        "particle_update_counts": update_counts,
        "particle_mean_final_inlier_probability": final_inlier_probabilities,
        "maximum_pairwise_endpoint_rmse_m": max(pairwise_rmse, default=0.0),
    }
    diagnostics.update(metadata or {})
    diagnostics["belief_provider"] = belief_provider_manifest.as_dict()
    identifiers = particle_ids or tuple(
        f"theta_{index:04d}" for index in range(particle_count)
    )
    return TwinBelief(
        context=context,
        endpoint_frame=endpoint,
        particle_ids=identifiers,
        theta_names=theta_names,
        endpoint_position_m=endpoint_positions,
        endpoint_velocity_mps=endpoint_velocities,
        theta=particle_values,
        discrepancy_mean_m=discrepancy_means,
        discrepancy_variance_m2=discrepancy_variances,
        weights=particle_weights,
        metadata=diagnostics,
    )


def export_official_phystwin_twin_belief(
    backend: OfficialPhysTwinBackend,
    *,
    context: CausalContext,
    config: BPTBeliefExportConfig | None = None,
) -> TwinBelief:
    """Replay every selected theta particle through O- in official Warp."""

    if context.case_id != backend.case_name:
        raise ValueError("causal context case does not match the PhysTwin backend")
    if (
        context.o_minus.frame_start != 0
        or context.o_minus.frame_stop != backend.train_end_frame
    ):
        raise ValueError("causal context O- does not match the backend training split")
    self_collision = (
        released_self_collision_for_case(backend.case_name)
        if backend.config.self_collision is None
        else backend.config.self_collision
    )
    simulator_configuration_id = backend.replay_simulator_configuration_id(
        backend.graph
    )
    released_initial_state_id = backend.replay_released_initial_state_id()
    replay_provider: PhysTwinReplayProvider = create_official_replay_provider(
        backend.official_repo,
        backend.data,
        backend.optimal,
        backend.checkpoint_path,
        backend.graph,
        num_surface_points=backend.original_count + len(backend.surface_points),
        original_count=backend.original_count,
        dt=backend.config.dt,
        num_substeps=backend.config.num_substeps,
        self_collision=bool(self_collision),
        simulator_configuration_id=simulator_configuration_id,
        released_initial_state_id=released_initial_state_id,
        deterministic_spring_forces=backend.config.deterministic_spring_forces,
        spring_parameterization="grouped",
        device=backend.config.device,
    )
    replay_positions = []
    replay_velocities = []
    replay_records: list[dict[str, Any]] = []
    try:
        for particle_index, particle in enumerate(backend.particles.log_scales):
            request_id = stable_replay_identifier(
                "causal4d-initial-replay-request-v1",
                {
                    "simulator_configuration_id": simulator_configuration_id,
                    "initial_state_id": released_initial_state_id,
                    "particle_index": particle_index,
                    "group_log_scales_sha256": array_sha256(particle),
                    "controller_points_sha256": array_sha256(backend.controller_points),
                    "frame_count": backend.train_end_frame,
                },
            )
            request = InitialReplayRequestV1(
                request_id=request_id,
                simulator_configuration_id=simulator_configuration_id,
                initial_state_id=released_initial_state_id,
                group_log_scales=particle,
                controller_points_m=backend.controller_points,
                frame_count=backend.train_end_frame,
            )
            replay = replay_provider.replay(request)
            validate_replay_trajectory(
                request,
                replay,
                expected_dt_s=backend.frame_dt_s,
            )
            replay_positions.append(replay.positions_m)
            replay_velocities.append(replay.velocities_mps)
            replay_records.append(
                {
                    "request_id": request_id,
                    "simulator_configuration_id": simulator_configuration_id,
                    "initial_state_id": released_initial_state_id,
                    "particle_index": particle_index,
                    "frame_ids": replay.frame_ids.tolist(),
                    "dt_s": float(replay.dt_s),
                    "positions_sha256": array_sha256(replay.positions_m),
                    "velocities_sha256": array_sha256(replay.velocities_mps),
                }
            )
    finally:
        replay_provider.close()

    valid = target_validity(backend.visible, backend.motion_valid)
    particle_ids = tuple(
        "grid_" + "_".join(map(str, grid_index))
        for grid_index in backend.particles.grid_indices
    )
    return build_twin_belief_from_replays(
        context=context,
        replay_positions_m=np.stack(replay_positions),
        replay_velocities_mps=np.stack(replay_velocities),
        observed_positions_m=backend.object_points,
        observed_valid=valid,
        theta=backend.particles.log_scales,
        theta_names=("object_spring_log_scale", "controller_spring_log_scale"),
        weights=backend.particles.weights,
        particle_ids=particle_ids,
        metadata={
            "profile_path": str(backend.profile_path.resolve()),
            "profile_weight_key": backend.particles.source_weight_key,
            "profile_support_method": backend.particles.selection_method,
            "profile_retained_probability_mass": backend.particles.retained_probability_mass,
            "profile_represented_probability_mass": (
                backend.particles.represented_probability_mass
            ),
            "official_backend": backend.default_manifest(),
            "replay_provider_api_version": 2,
            "initial_replay_requests": replay_records,
        },
        config=config,
    )
