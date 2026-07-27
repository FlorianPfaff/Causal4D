from pathlib import Path

import numpy as np

from bayesian_phystwin.causal4d_graph_provider_v1 import PhysTwinSpringGraph
from causal4d.contracts import TwinBelief, array_sha256
from causal4d.graph_provider_contract import require_bayesian_phystwin_graph_provider
from causal4d.phystwin_backend import (
    BayesianPhysTwinParticles,
    OfficialPhysTwinBackend,
    OfficialPhysTwinBackendConfig,
    PhysTwinActionProposal,
    PhysTwinContactState,
    PhysTwinHypothesisConfig,
    build_contact_states,
    hidden_action_proposals,
    load_bayesian_phystwin_particles,
    shift_phystwin_attachment_graph,
    transform_controller_trajectory,
)
from causal4d.provider_contract import require_bayesian_phystwin_provider


def test_profile_loader_selects_and_renormalizes_high_mass_particles(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.npz"
    np.savez(
        profile,
        object_log_scales=np.asarray([-0.2, 0.2]),
        controller_log_scales=np.asarray([-0.1, 0.1]),
        posterior_weights=np.asarray([[0.1, 0.2], [0.6, 0.1]]),
    )
    particles = load_bayesian_phystwin_particles(profile, maximum_count=2)
    assert np.allclose(particles.log_scales[0], [0.2, -0.1])
    assert np.allclose(particles.weights, [0.75, 0.25])
    assert np.isclose(particles.retained_probability_mass, 0.8)

    coreset = load_bayesian_phystwin_particles(
        profile,
        maximum_count=2,
        support_method="weighted_coreset",
    )
    assert coreset.selection_method == "weighted_coreset"
    assert coreset.represented_probability_mass == 1.0
    assert coreset.source_particle_count == 4
    assert np.isclose(np.sum(coreset.weights), 1.0)


def test_profile_loader_preserves_staged_probability_mass(tmp_path: Path) -> None:
    profile = tmp_path / "truncated_profile.npz"
    source_weights = np.asarray([[0.5, 0.3], [0.1, 0.1]])
    prediction_weights = np.asarray([[5 / 9, 3 / 9], [1 / 9, 0.0]])
    np.savez(
        profile,
        object_log_scales=np.asarray([-0.2, 0.2]),
        controller_log_scales=np.asarray([-0.1, 0.1]),
        posterior_weights=np.full((2, 2), 0.25),
        source_prediction_weights=source_weights,
        prediction_weights=prediction_weights,
    )

    full = load_bayesian_phystwin_particles(profile, maximum_count=10)
    assert len(full.weights) == 3
    assert [1, 1] not in full.grid_indices.tolist()
    assert full.profile_grid_cell_count == 4
    assert full.source_particle_count == 3
    assert np.isclose(full.bpt_retained_probability_mass, 0.9)
    assert np.isclose(full.causal4d_retained_probability_mass, 1.0)
    assert np.isclose(full.retained_probability_mass, 0.9)

    reduced = load_bayesian_phystwin_particles(profile, maximum_count=2)
    np.testing.assert_array_equal(reduced.grid_indices, [[0, 0], [0, 1]])
    np.testing.assert_allclose(reduced.weights, [5 / 8, 3 / 8])
    assert reduced.bpt_source_weight_key == "source_prediction_weights"
    assert np.isclose(reduced.bpt_retained_probability_mass, 0.9)
    assert np.isclose(reduced.causal4d_retained_probability_mass, 8 / 9)
    assert np.isclose(reduced.retained_probability_mass, 0.8)

    coreset = load_bayesian_phystwin_particles(
        profile,
        maximum_count=2,
        support_method="weighted_coreset",
    )
    assert np.isclose(coreset.bpt_retained_probability_mass, 0.9)
    assert np.isclose(coreset.causal4d_represented_probability_mass, 1.0)
    assert np.isclose(coreset.represented_probability_mass, 0.9)
    assert [1, 1] not in coreset.grid_indices.tolist()

    accounting = reduced.probability_mass_accounting()
    assert np.isclose(
        accounting["bpt_truncation"]["retained_probability_mass"],
        0.9,
    )
    assert np.isclose(
        accounting["causal4d_support_reduction"][
            "directly_retained_probability_mass"
        ],
        8 / 9,
    )
    assert np.isclose(
        accounting["composed_relative_to_original_posterior"][
            "directly_retained_probability_mass"
        ],
        0.8,
    )

    backend = object.__new__(OfficialPhysTwinBackend)
    backend.provider_manifest = require_bayesian_phystwin_provider()
    backend.graph_provider_manifest = require_bayesian_phystwin_graph_provider()
    backend.case_name = "unit_case"
    backend.train_end_frame = 3
    backend.official_repo = tmp_path / "official"
    backend.final_data_path = tmp_path / "final_data.pkl"
    backend.optimal_params_path = tmp_path / "optimal_params.pkl"
    backend.checkpoint_path = tmp_path / "checkpoint.pt"
    backend.baseline_trajectory_path = tmp_path / "baseline.pkl"
    backend.profile_path = profile
    backend.particles = reduced
    backend.controller_groups = np.asarray([0])
    backend.config = OfficialPhysTwinBackendConfig()
    manifest = backend.default_manifest()["parameter_particles"]
    assert np.isclose(manifest["retained_probability_mass"], 0.8)
    assert manifest["probability_mass_accounting"] == accounting


def test_hidden_action_proposals_never_read_withheld_future() -> None:
    controls = np.zeros((12, 3, 3), dtype=float)
    controls[:6, :, 0] = np.arange(6)[:, None] * 0.01
    changed = controls.copy()
    changed[6:] = 1000.0
    first = hidden_action_proposals(controls, start_frame=6)
    second = hidden_action_proposals(changed, start_frame=6)
    assert [value.proposal_id for value in first] == [
        value.proposal_id for value in second
    ]
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.controller_points_m, right.controller_points_m)


def test_contact_beam_preserves_every_latent_channel() -> None:
    states = build_contact_states(
        2,
        PhysTwinHypothesisConfig(maximum_contact_states=14),
    )
    assert np.isclose(sum(state.prior_weight for state in states), 1.0)
    assert {-1, 0, 1} <= {
        shift for state in states for shift in state.attachment_shifts
    }
    assert {0, 2} <= {state.delay_steps for state in states}
    assert {0.0, 0.2} <= {state.slip_fraction for state in states}
    assert {-8.0, 0.0, 8.0} <= {state.rotation_degrees for state in states}
    assert {0.85, 1.0, 1.15} <= {state.gain_multiplier for state in states}


def test_attachment_shift_is_one_hop_and_keeps_spring_contract() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    springs = np.asarray([[0, 1], [1, 2], [2, 3], [4, 1]], dtype=np.int32)
    rest = np.linalg.norm(vertices[springs[:, 0]] - vertices[springs[:, 1]], axis=1)
    graph = PhysTwinSpringGraph(
        vertices=vertices,
        springs=springs,
        rest_lengths=rest.astype(np.float32),
        masses=np.ones(5, dtype=np.float32),
        num_object_springs=3,
    )
    positive = shift_phystwin_attachment_graph(graph, np.asarray([0]), (1,))
    negative = shift_phystwin_attachment_graph(graph, np.asarray([0]), (-1,))
    nominal = shift_phystwin_attachment_graph(graph, np.asarray([0]), (0,))
    assert len(positive.graph.springs) == len(graph.springs)
    assert positive.graph.springs[-1].tolist() == [4, 2]
    assert negative.graph.springs[-1].tolist() == [4, 0]
    assert np.array_equal(nominal.graph.springs, graph.springs)


def test_controller_transform_applies_future_delay_and_slip_only() -> None:
    controls = np.zeros((6, 1, 3), dtype=float)
    controls[:, 0, 0] = np.arange(6)
    state = PhysTwinContactState(
        attachment_shifts=(0,),
        gain_multiplier=1.0,
        delay_steps=1,
        slip_fraction=0.5,
        rotation_degrees=0.0,
        prior_weight=1.0,
    )
    transformed = transform_controller_trajectory(
        controls, np.asarray([0]), state, start_frame=3
    )
    assert np.array_equal(transformed[:3], controls[:3])
    assert np.isclose(transformed[3, 0, 0], controls[2, 0, 0])
    assert np.isclose(transformed[4, 0, 0], 2.5)


def test_backend_context_identifies_the_ordered_counterfactual_library() -> None:
    backend = object.__new__(OfficialPhysTwinBackend)
    backend.case_name = "unit_case"
    backend.train_end_frame = 3
    backend.frame_count = 6
    backend.object_points = np.zeros((6, 2, 3), dtype=float)
    backend.controller_points = np.zeros((6, 1, 3), dtype=float)
    first = PhysTwinActionProposal(
        proposal_id="first",
        controller_points_m=backend.controller_points.copy(),
        prior_weight=0.5,
        future_action_observed=False,
        provenance="unit",
    )
    second_controls = backend.controller_points.copy()
    second_controls[3:, 0, 0] = 1.0
    second = PhysTwinActionProposal(
        proposal_id="second",
        controller_points_m=second_controls,
        prior_weight=0.5,
        future_action_observed=False,
        provenance="unit",
    )
    context = backend.causal_context((first, second), protocol_id="unit")
    expected = np.stack([first.controller_points_m[3:], second.controller_points_m[3:]])
    assert context.u_cf.action_id == "action_library[first,second]"
    assert context.u_cf.trajectory_sha256 == array_sha256(expected)


def test_backend_reuses_factual_belief_for_a_new_counterfactual_query() -> None:
    backend = object.__new__(OfficialPhysTwinBackend)
    backend.case_name = "unit_case"
    backend.train_end_frame = 3
    backend.frame_count = 6
    backend.object_points = np.zeros((6, 2, 3), dtype=float)
    backend.controller_points = np.zeros((6, 1, 3), dtype=float)
    backend.baseline = np.zeros((6, 4, 3), dtype=float)
    backend.particles = BayesianPhysTwinParticles(
        log_scales=np.asarray([[0.0, 0.0], [0.2, -0.1]]),
        weights=np.asarray([0.6, 0.4]),
        grid_indices=np.asarray([[0, 0], [1, 0]]),
        source_weight_key="posterior_weights",
        retained_probability_mass=1.0,
    )
    proposal = PhysTwinActionProposal(
        proposal_id="future",
        controller_points_m=backend.controller_points.copy(),
        prior_weight=1.0,
        future_action_observed=False,
        provenance="unit",
    )
    context = backend.causal_context((proposal,), protocol_id="unit")
    endpoints = np.zeros((2, 4, 3), dtype=float)
    endpoints[1, :, 0] = 0.01
    belief = TwinBelief(
        context=context,
        endpoint_frame=2,
        particle_ids=("p0", "p1"),
        theta_names=("object", "controller"),
        endpoint_position_m=endpoints,
        endpoint_velocity_mps=np.zeros_like(endpoints),
        theta=backend.particles.log_scales,
        discrepancy_mean_m=np.zeros_like(endpoints),
        discrepancy_variance_m2=np.ones_like(endpoints) * 1e-5,
        weights=backend.particles.weights,
    )
    backend._validate_twin_belief(belief, (proposal,))
    changed = PhysTwinActionProposal(
        proposal_id="changed",
        controller_points_m=backend.controller_points.copy(),
        prior_weight=1.0,
        future_action_observed=False,
        provenance="unit",
    )
    backend._validate_twin_belief(belief, (changed,))
