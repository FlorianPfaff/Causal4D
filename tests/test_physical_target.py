from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from causal4d.contracts import PhysicalPosterior, array_sha256, build_causal_context
from causal4d.physical_target import (
    PhysicalTargetBundle,
    build_physical_target,
    load_physical_target,
    save_physical_target,
)


def _fixture() -> tuple[PhysicalPosterior, np.ndarray, np.ndarray]:
    observations = np.arange(7 * 2 * 3, dtype=np.float32).reshape(7, 2, 3)
    actions = np.zeros((7, 1, 3), dtype=np.float64)
    context = build_causal_context(
        protocol_id="physical-target-unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    states = np.zeros((2, 5, 2, 3), dtype=np.float32)
    posterior = PhysicalPosterior(
        context=context,
        component_ids=("a", "b"),
        state_trajectories_m=states,
        readout_trajectories_m=states,
        readout_variance_m2=np.full((2, 2, 3), 1e-5, dtype=np.float32),
        weights=np.asarray([0.75, 0.25]),
        phi=np.asarray([[1.0], [1.0]]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )
    valid = np.ones(observations.shape[:2], dtype=bool)
    valid[5, 1] = False
    return posterior, observations, valid


def _bundle() -> tuple[PhysicalTargetBundle, PhysicalPosterior, np.ndarray]:
    posterior, observations, valid = _fixture()
    bundle = build_physical_target(
        posterior.context,
        observations,
        valid,
        source_final_data_sha256="a" * 64,
        metadata={"producer": "unit-test"},
    )
    return bundle, posterior, observations


def test_physical_target_round_trip_preserves_canonical_backend_bytes(
    tmp_path: Path,
) -> None:
    bundle, posterior, observations = _bundle()
    path = tmp_path / "target.npz"
    save_physical_target(path, bundle)
    restored = load_physical_target(path)

    assert restored.artifact_id == bundle.artifact_id
    assert restored.object_points.dtype == observations.dtype
    assert np.array_equal(restored.object_points, observations[2:])
    assert not restored.object_points.flags.writeable
    truth, valid = restored.aligned_for_posterior(posterior)
    assert truth.shape == posterior.readout_trajectories_m.shape[1:]
    assert valid.shape == truth.shape[:2]


def test_evaluation_target_hashes_the_exact_selected_suffix() -> None:
    bundle, _, observations = _bundle()
    target = bundle.evaluation_target(start_frame=2)
    assert target.target.frame_start == bundle.anchor_frame + 2
    assert target.target.frame_stop == bundle.context.o_plus.frame_stop
    assert target.target.content_sha256 == array_sha256(observations[4:])


def test_builder_reproduces_backend_float32_observation_semantics() -> None:
    bundle, posterior, observations = _bundle()
    valid = np.ones(observations.shape[:2], dtype=bool)
    valid[5, 1] = False
    rebuilt = build_physical_target(
        posterior.context,
        observations.astype(np.float64),
        valid,
        source_final_data_sha256="b" * 64,
    )
    assert rebuilt.object_points.dtype == np.dtype(np.float32)
    assert np.array_equal(rebuilt.object_points, observations[2:])
    assert rebuilt.context.o_plus.content_sha256 == bundle.context.o_plus.content_sha256


def test_physical_target_rejects_tampered_payload(tmp_path: Path) -> None:
    bundle, _, _ = _bundle()
    path = tmp_path / "target.npz"
    save_physical_target(path, bundle)
    with np.load(path, allow_pickle=False) as archive:
        descriptor = np.asarray(archive["descriptor_json"])
        points = np.asarray(archive["object_points"]).copy()
        valid = np.asarray(archive["validity_mask"])
    points[1, 0, 0] += np.float32(1.0)
    np.savez_compressed(
        path,
        descriptor_json=descriptor,
        object_points=points,
        validity_mask=valid,
    )
    with pytest.raises(ValueError, match=r"O\+ digest|payload hashes"):
        load_physical_target(path)


def test_physical_target_rejects_context_or_support_mismatch() -> None:
    bundle, posterior, observations = _bundle()
    actions = np.zeros((7, 1, 3), dtype=np.float64)
    changed_context = build_causal_context(
        protocol_id="other-protocol",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    changed_posterior = replace(posterior, context=changed_context)
    with pytest.raises(ValueError, match="context does not match"):
        bundle.aligned_for_posterior(changed_posterior)

    too_short = replace(
        posterior,
        state_trajectories_m=posterior.state_trajectories_m[:, :-1],
        readout_trajectories_m=posterior.readout_trajectories_m[:, :-1],
    )
    with pytest.raises(ValueError, match="frame count"):
        bundle.aligned_for_posterior(too_short)


def test_physical_target_publication_is_exactly_once_by_default(tmp_path: Path) -> None:
    bundle, _, _ = _bundle()
    path = tmp_path / "target.npz"
    save_physical_target(path, bundle)
    with pytest.raises(FileExistsError):
        save_physical_target(path, bundle)
    save_physical_target(path, bundle, overwrite=True)


def test_physical_target_rejects_coercion_dependent_arrays() -> None:
    bundle, _, _ = _bundle()
    with pytest.raises(ValueError, match="canonical float32"):
        PhysicalTargetBundle(
            context=bundle.context,
            object_points=bundle.object_points.astype(np.int64),
            validity_mask=bundle.validity_mask,
            source_final_data_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="boolean dtype"):
        PhysicalTargetBundle(
            context=bundle.context,
            object_points=bundle.object_points,
            validity_mask=bundle.validity_mask.astype(np.uint8),
            source_final_data_sha256="a" * 64,
        )
