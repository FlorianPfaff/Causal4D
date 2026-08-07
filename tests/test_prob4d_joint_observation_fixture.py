from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from causal4d.joint_observation import joint_component_log_likelihoods
from causal4d.prob4d_joint_observation import joint_observation_from_prob4d


FIXTURE = Path(__file__).parent / "fixtures" / "prob4d_joint_observation_v1.json"


def _fixture() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    arrays = {
        name: np.asarray(entry["values"], dtype=np.dtype(entry["dtype"]))
        for name, entry in payload["arrays"].items()
    }
    return payload["descriptor"], arrays


def test_prob4d_joint_fixture_runs_through_full_covariance_likelihood() -> None:
    descriptor, arrays = _fixture()
    evidence, diagnostics = joint_observation_from_prob4d(
        descriptor,
        arrays,
        rollout_frame_ids=(0, 1, 2, 3, 4, 5),
        entity_to_node={0: 0, 1: 1},
        reliability_policy="record_only",
    )

    assert diagnostics.provider_validation["validated"] is True
    assert diagnostics.row_count == 6
    assert diagnostics.observation_count == 18
    assert diagnostics.factor_rank == 5
    assert evidence.base_covariance_representation == "block_diagonal"
    assert evidence.base_covariance_m2.shape == (6, 3, 3)
    assert evidence.shared_covariance_factor_m.shape == (18, 5)

    shared_covariance = (
        evidence.shared_covariance_factor_m
        @ evidence.shared_covariance_factor_m.T
    )
    assert shared_covariance[0, 6] != 0.0

    components = np.zeros((2, 6, 2, 3), dtype=float)
    for row, (frame, entity) in enumerate(
        zip(arrays["frame_ids"], arrays["entity_ids"], strict=True)
    ):
        components[0, int(frame), int(entity)] = arrays["mean_xyz_m"][row]
        components[1, int(frame), int(entity)] = (
            arrays["mean_xyz_m"][row] + np.array([0.01, -0.005, 0.002])
        )

    score, likelihood_diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=6,
    )

    assert np.all(np.isfinite(score))
    assert score[0] > score[1]
    assert likelihood_diagnostics.base_covariance_representation == "block_diagonal"
    assert likelihood_diagnostics.used_low_rank_path is True
