from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.held_out_target import (
    HeldOutPhysicalTarget,
    load_held_out_physical_target,
    save_held_out_physical_target,
)


def _target() -> HeldOutPhysicalTarget:
    observations = np.zeros((7, 1, 3), dtype=float)
    actions = np.zeros((7, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="held-out-target-archive-members",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    states = np.zeros((1, 5, 1, 3), dtype=float)
    posterior = PhysicalPosterior(
        context=context,
        component_ids=("component",),
        state_trajectories_m=states,
        readout_trajectories_m=states,
        readout_variance_m2=np.full((1, 1, 3), 1e-5),
        weights=np.asarray([1.0]),
        phi=np.asarray([[1.0]]),
        kappa_cf=np.asarray([[0.0]]),
        hypothesis_indices=np.asarray([0]),
        twin_particle_indices=np.asarray([0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )
    return HeldOutPhysicalTarget(
        context=context,
        source_query_id=posterior.source_query_id,
        trajectory_frame_start=context.o_minus.frame_stop - 1,
        node_indices=np.asarray([0], dtype=np.int64),
        positions_m=states[0],
        validity_mask=np.ones(states.shape[1:3], dtype=bool),
        source_kind="synthetic_test_target",
        source_revision="test-v1",
        source_content_sha256="4" * 64,
    )


def test_held_out_target_rejects_undeclared_zip_members(tmp_path: Path) -> None:
    path = tmp_path / "target.npz"
    save_held_out_physical_target(path, _target())
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("unexpected.txt", "not part of the target schema")
    with pytest.raises(ValueError, match="ZIP members"):
        load_held_out_physical_target(path)
