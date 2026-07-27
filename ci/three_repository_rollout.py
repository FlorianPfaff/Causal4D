"""Causal4D lineage, mass accounting, and fake replay golden-path checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from bayesian_phystwin.causal4d_provider_v2 import (
    PhysTwinReplayProvider,
    ReplayRequestV1,
    ReplayTrajectoryV1,
    RestartReplayRequestV1,
)

from three_repository_common import (
    EXPECTED_OBSERVATION_ARTIFACT_ID,
    array_digest,
    require,
)


class _FakeReplayProvider:
    def __init__(self, **kwargs: Any) -> None:
        self.device = str(kwargs["device"])
        self.frame_dt_s = float(kwargs["dt"]) * int(kwargs["num_substeps"])
        self.simulator_configuration_id = str(kwargs["simulator_configuration_id"])
        self.released_initial_state_id = str(kwargs["released_initial_state_id"])
        self.requests: list[RestartReplayRequestV1] = []
        self.restart_calls = 0
        self.closed = False

    def replay(self, request: ReplayRequestV1) -> ReplayTrajectoryV1:
        require(not self.closed, "fake provider is closed")
        require(
            isinstance(request, RestartReplayRequestV1),
            "golden rollout did not use a restart replay-v2 request",
        )
        position = np.asarray(request.position_m, dtype=np.float32).copy()
        velocity = np.asarray(request.velocity_mps, dtype=np.float32).copy()
        require(position.shape == velocity.shape, "restart state shapes differ")
        frame_count = request.stop_frame - request.start_frame
        require(frame_count > 0, "restart interval is empty")
        positions = np.empty((frame_count, *position.shape), dtype=np.float32)
        velocities = np.empty_like(positions)
        scale = float(np.sum(np.exp(request.group_log_scales)))
        for offset in range(frame_count):
            position = position + 0.01 * velocity
            position[:, 0] += np.float32(0.0005 * scale)
            position[:, 1] += np.float32(0.0001 * (offset + 1))
            position[-1] = request.controller_points_m[request.start_frame + offset, 0]
            velocity[:, 0] += np.float32(0.00001 * (offset + 1))
            positions[offset] = position
            velocities[offset] = velocity
        self.requests.append(request)
        self.restart_calls += 1
        return ReplayTrajectoryV1(
            positions_m=positions,
            velocities_mps=velocities,
            frame_ids=np.arange(request.start_frame, request.stop_frame),
            dt_s=self.frame_dt_s,
            request_id=request.request_id,
            simulator_configuration_id=request.simulator_configuration_id,
            initial_state_id=request.initial_state_id,
        )

    def close(self) -> None:
        self.closed = True


def _profile_particles(workdir: Path) -> tuple[Any, Path]:
    from causal4d.phystwin_backend import load_bayesian_phystwin_particles

    profile_path = workdir / "parameter-profile.npz"
    np.savez_compressed(
        profile_path,
        object_log_scales=np.asarray([-0.2, 0.2]),
        controller_log_scales=np.asarray([-0.1, 0.1]),
        posterior_weights=np.full((2, 2), 0.25),
        source_prediction_weights=np.asarray([[0.5, 0.3], [0.1, 0.1]]),
        prediction_weights=np.asarray([[5 / 9, 3 / 9], [1 / 9, 0.0]]),
    )
    particles = load_bayesian_phystwin_particles(
        profile_path,
        maximum_count=2,
    )
    require(
        [1, 1] not in particles.grid_indices.tolist(),
        "zero-mass cell selected",
    )
    require(
        np.isclose(particles.bpt_retained_probability_mass, 0.9),
        "BPT retained mass changed",
    )
    require(
        np.isclose(particles.causal4d_retained_probability_mass, 8 / 9),
        "Causal4D retained mass changed",
    )
    require(
        np.isclose(particles.retained_probability_mass, 0.8),
        "composed retained mass changed",
    )
    return particles, profile_path


def _invalid_composed_mass_rejection() -> dict[str, str]:
    from causal4d.phystwin_backend import BayesianPhysTwinParticles

    try:
        BayesianPhysTwinParticles(
            log_scales=np.asarray([[0.0, 0.0]]),
            weights=np.asarray([1.0]),
            grid_indices=np.asarray([[0, 0]]),
            source_weight_key="prediction_weights",
            retained_probability_mass=0.7,
            bpt_retained_probability_mass=0.9,
            causal4d_retained_probability_mass=8 / 9,
        )
    except ValueError as error:
        return {
            "label": "posterior-mass:incorrect-composition",
            "error": type(error).__name__,
            "message": str(error),
        }
    raise RuntimeError("incorrect composed posterior mass was accepted")


def _minimal_backend(
    *,
    particles: Any,
    profile_path: Path,
    provider_manifest: Any,
    replay_provider_manifest: Any,
    graph_provider_manifest: Any,
    workdir: Path,
) -> tuple[Any, np.ndarray, np.ndarray]:
    from causal4d.phystwin_backend import (
        OfficialPhysTwinBackend,
        OfficialPhysTwinBackendConfig,
    )

    backend = object.__new__(OfficialPhysTwinBackend)
    backend.provider_manifest = provider_manifest
    backend.replay_provider_manifest = replay_provider_manifest
    backend.graph_provider_manifest = graph_provider_manifest
    backend.official_repo = workdir / "official-phystwin"
    backend.final_data_path = workdir / "final-data.pkl"
    backend.optimal_params_path = workdir / "optimal.pkl"
    backend.checkpoint_path = workdir / "checkpoint.pt"
    backend.baseline_trajectory_path = workdir / "baseline.pkl"
    backend.profile_path = profile_path
    backend.case_name = "joint-gauge-contract"
    backend.train_end_frame = 6
    backend.frame_count = 9
    backend.original_count = 2
    backend.hand_count = 1
    backend.surface_points = np.zeros((0, 3), dtype=np.float32)
    backend.data = {}
    backend.optimal = {}
    backend.object_points = np.zeros((9, 2, 3), dtype=np.float32)
    backend.visible = np.ones((9, 2), dtype=bool)
    backend.motion_valid = np.ones((9, 2), dtype=bool)
    controls = np.zeros((9, 1, 3), dtype=np.float32)
    controls[6:, 0, 0] = np.asarray([0.01, 0.02, 0.03], dtype=np.float32)
    backend.controller_points = controls
    backend.controller_groups = np.asarray([0], dtype=np.int64)
    backend.particles = particles
    backend.baseline = np.zeros((9, 3, 3), dtype=np.float32)
    backend.config = OfficialPhysTwinBackendConfig(
        dt=0.01,
        num_substeps=1,
        self_collision=False,
        device="cpu",
    )
    backend.source_artifacts_sha256 = {}
    backend.base_simulator_configuration_id = "golden-base-configuration-v1"
    backend.released_initial_state_id = "golden-released-state-v1"

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    springs = np.asarray([[0, 1], [2, 1]], dtype=np.int32)
    backend.graph = SimpleNamespace(
        vertices=vertices,
        springs=springs,
        rest_lengths=np.linalg.norm(
            vertices[springs[:, 0]] - vertices[springs[:, 1]],
            axis=1,
        ).astype(np.float32),
        masses=np.ones(3, dtype=np.float32),
        num_object_springs=1,
        num_object_points=2,
    )
    return backend, vertices, controls


def run_causal4d_rollout(
    observation_path: Path,
    bpt_result: Any,
    bpt_summary: dict[str, Any],
    workdir: Path,
) -> dict[str, Any]:
    """Validate lineage and execute one deterministic fake-provider query."""

    import causal4d.phystwin_backend as backend_module
    from causal4d.contracts import TwinBelief, load_contract, save_contract
    from causal4d.observation_lineage import (
        bind_twin_belief_observation_lineage,
        load_observation_lineage,
        validate_twin_belief_observation_lineage,
    )
    from causal4d.phystwin_backend import (
        PhysTwinActionProposal,
        PhysTwinHypothesisConfig,
        load_rollout_bank,
        save_rollout_bank,
    )
    from causal4d.graph_provider_contract import (
        require_bayesian_phystwin_graph_provider,
    )
    from causal4d.provider_contract import (
        require_bayesian_phystwin_provider,
        validate_bayesian_phystwin_provider,
    )
    from causal4d.replay_provider_contract import (
        require_bayesian_phystwin_replay_provider,
        validate_bayesian_phystwin_replay_provider,
    )

    provider_manifest = require_bayesian_phystwin_provider(
        provider_revision="installed-wheel-golden-path"
    )
    compatibility = validate_bayesian_phystwin_provider(provider_manifest)
    require(compatibility.compatible, "installed BPT provider is incompatible")
    replay_provider_manifest = require_bayesian_phystwin_replay_provider(
        provider_revision="installed-wheel-golden-path"
    )
    replay_compatibility = validate_bayesian_phystwin_replay_provider(
        replay_provider_manifest
    )
    require(
        replay_compatibility.compatible,
        "installed BPT replay-v2 provider is incompatible",
    )
    graph_provider_manifest = require_bayesian_phystwin_graph_provider(
        provider_revision="installed-wheel-golden-path"
    )

    lineage = load_observation_lineage(observation_path)
    require(
        lineage.artifact_id == EXPECTED_OBSERVATION_ARTIFACT_ID,
        "Causal4D independently computed a different observation ID",
    )
    require(
        lineage.provider_validation.get("stream_contract_version") == 2,
        "Causal4D did not resolve the joint-gauge stream as contract v2",
    )
    particles, profile_path = _profile_particles(workdir)
    backend, vertices, controls = _minimal_backend(
        particles=particles,
        profile_path=profile_path,
        provider_manifest=provider_manifest,
        replay_provider_manifest=replay_provider_manifest,
        graph_provider_manifest=graph_provider_manifest,
        workdir=workdir,
    )

    proposal = PhysTwinActionProposal(
        proposal_id="golden-counterfactual",
        controller_points_m=controls,
        prior_weight=1.0,
        future_action_observed=False,
        provenance="deterministic installed-wheel golden path",
    )
    context = backend.causal_context(
        (proposal,),
        protocol_id="three-repository-installed-wheel-v1",
    )
    endpoints = np.repeat(
        vertices[None],
        len(particles.weights),
        axis=0,
    ).astype(float)
    coefficient = float(bpt_result.state_coefficients[0])
    endpoints[:, 0, 0] += coefficient
    endpoints[:, 1, 0] -= coefficient
    velocities = np.zeros_like(endpoints)
    twin_belief = TwinBelief(
        context=context,
        endpoint_frame=5,
        particle_ids=tuple(
            f"particle-{index}" for index in range(len(particles.weights))
        ),
        theta_names=(
            "object_spring_log_scale",
            "controller_spring_log_scale",
        ),
        endpoint_position_m=endpoints,
        endpoint_velocity_mps=velocities,
        theta=particles.log_scales,
        discrepancy_mean_m=np.zeros_like(endpoints),
        discrepancy_variance_m2=np.full_like(endpoints, 1e-5),
        weights=particles.weights,
        metadata={
            "bpt_update_id": bpt_summary["update_id"],
            "bpt_state_coefficient_m": coefficient,
            "provider_manifest_id": provider_manifest.manifest_id,
        },
    )
    bound_belief = bind_twin_belief_observation_lineage(twin_belief, lineage)
    validation = validate_twin_belief_observation_lineage(
        bound_belief,
        lineage,
    )
    require(validation["lineage_bound"] is True, "TwinBelief lineage was not bound")
    twin_path = workdir / "twin-belief.npz"
    save_contract(twin_path, bound_belief)
    restored_twin = load_contract(twin_path)
    require(
        restored_twin.artifact_id == bound_belief.artifact_id,
        "TwinBelief round trip changed its ID",
    )

    providers: list[_FakeReplayProvider] = []

    def provider_factory(*args: Any, **kwargs: Any) -> _FakeReplayProvider:
        del args
        require(
            kwargs.get("device") == "cpu",
            "golden provider was not CPU-only",
        )
        provider = _FakeReplayProvider(**kwargs)
        require(
            isinstance(provider, PhysTwinReplayProvider),
            "fake provider does not satisfy the public replay-v2 protocol",
        )
        providers.append(provider)
        return provider

    original_factory = backend_module.create_official_replay_provider
    backend_module.create_official_replay_provider = provider_factory
    try:
        bank, manifest = backend.build_rollout_bank(
            (proposal,),
            twin_belief=restored_twin,
            hypothesis_config=PhysTwinHypothesisConfig(
                attachment_shift_values=(0,),
                gain_values=(1.0,),
                delay_values=(0,),
                slip_values=(0.0,),
                rotation_values_degrees=(0.0,),
                maximum_contact_states=1,
            ),
        )
    finally:
        backend_module.create_official_replay_provider = original_factory

    require(len(providers) == 1, "golden rollout created unexpected providers")
    require(providers[0].closed, "fake provider was not closed")
    require(providers[0].restart_calls == 2, "unexpected replay count")
    require(len(providers[0].requests) == 2, "zero-mass support was replayed")
    require(
        all(
            request.simulator_configuration_id
            == providers[0].simulator_configuration_id
            for request in providers[0].requests
        ),
        "replay request configuration identity changed",
    )
    require(
        all(
            request.velocity_mps.shape == request.position_m.shape
            for request in providers[0].requests
        ),
        "restart request lost velocity history",
    )
    require(
        bank.trajectories.shape == (1, 2, 4, 2, 3),
        f"rollout bank shape changed: {bank.trajectories.shape}",
    )
    np.testing.assert_array_equal(
        bank.trajectories[0, :, 0],
        restored_twin.endpoint_position_m[:, :2].astype(np.float32),
    )
    require(
        np.any(bank.trajectories[:, :, 1:] != bank.trajectories[:, :, :-1]),
        "fake counterfactual rollout did not evolve",
    )
    accounting = particles.probability_mass_accounting()
    invalid_mass_rejection = _invalid_composed_mass_rejection()
    require(
        manifest["parameter_particles"]["probability_mass_accounting"] == accounting,
        "rollout manifest lost staged posterior-mass accounting",
    )

    rollout_path = workdir / "rollout-bank.npz"
    save_rollout_bank(rollout_path, bank, manifest)
    restored_bank, restored_manifest = load_rollout_bank(rollout_path)
    np.testing.assert_array_equal(restored_bank.trajectories, bank.trajectories)
    require(restored_manifest == manifest, "rollout manifest round trip changed")
    return {
        "provider_manifest_id": provider_manifest.manifest_id,
        "provider_compatibility": compatibility.as_dict(),
        "replay_provider_manifest_id": replay_provider_manifest.manifest_id,
        "replay_provider_compatibility": replay_compatibility.as_dict(),
        "observation_lineage": validation,
        "twin_belief_id": restored_twin.artifact_id,
        "twin_belief_path": str(twin_path),
        "rollout_bank_path": str(rollout_path),
        "rollout_digest": array_digest(restored_bank.trajectories),
        "rollout_shape": list(restored_bank.trajectories.shape),
        "restart_calls": providers[0].restart_calls,
        "replay_request_ids": [request.request_id for request in providers[0].requests],
        "replay_frame_ids": [
            list(range(request.start_frame, request.stop_frame))
            for request in providers[0].requests
        ],
        "probability_mass_accounting": accounting,
        "invalid_mass_rejection": invalid_mass_rejection,
        "zero_mass_cells_replayed": False,
    }
