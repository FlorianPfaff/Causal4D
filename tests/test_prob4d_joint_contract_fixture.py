from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from causal4d.observation_lineage import (
    bind_twin_belief_observation_lineage,
    compute_observation_artifact_id,
    load_observation_lineage,
)
from causal4d.prob4d_observation_lineage import PROB4D_JOINT_GAUGE_MODEL

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "prob4d_joint_observation_v1.json"
)


@dataclass(frozen=True)
class _Window:
    frame_stop: int
    frame_start: int = 0


@dataclass(frozen=True)
class _Context:
    case_id: str
    o_minus: _Window


@dataclass(frozen=True)
class _TwinBelief:
    context: _Context
    metadata: dict
    artifact_id: str = "f" * 64


def _parts() -> tuple[dict, dict[str, np.ndarray], str]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    descriptor = deepcopy(payload["descriptor"])
    arrays = {
        name: np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        for name, record in payload["arrays"].items()
    }
    return descriptor, arrays, payload["expected_artifact_id"]


def _write(path: Path, *, mutate=None) -> str:
    descriptor, arrays, expected = _parts()
    if mutate is not None:
        mutate(descriptor, arrays)
    artifact_id = compute_observation_artifact_id(descriptor, arrays)
    if mutate is None:
        assert artifact_id == expected
    descriptor["artifact_id"] = artifact_id
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
        **arrays,
    )
    return artifact_id


def test_joint_gauge_fixture_is_validated_and_bound(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    expected = _write(path)

    lineage = load_observation_lineage(path)
    twin = _TwinBelief(_Context("joint-gauge-contract", _Window(6)), {})
    bound = bind_twin_belief_observation_lineage(twin, lineage)

    assert lineage.artifact_id == expected
    assert (
        lineage.provider_validation["covariance_semantics"]
        == PROB4D_JOINT_GAUGE_MODEL
    )
    assert lineage.provider_validation["cross_window_covariance_preserved"] is True
    assert lineage.provider_validation["factor_group_count"] == 1
    assert lineage.provider_validation["factor_rank"] == 5
    assert bound.metadata[
        "source_observation_provider_validation"
    ] == lineage.provider_validation


def test_joint_gauge_fixture_rejects_per_window_factor_groups(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del descriptor
        arrays["factor_group_ids"] = arrays["window_indices"].copy()

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="one shared factor group"):
        load_observation_lineage(path)


def test_joint_gauge_fixture_rejects_rank_metadata_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del arrays
        descriptor["metadata"]["gauge_posterior"]["exported_factor_rank"] = 4

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="rank differs"):
        load_observation_lineage(path)


def test_joint_gauge_fixture_rejects_parent_order_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observation.npz"

    def mutate(descriptor, arrays):
        del arrays
        descriptor["metadata"]["gauge_posterior"]["parent_window_ids"][1] = (
            "window-2"
        )

    _write(path, mutate=mutate)
    with pytest.raises(ValueError, match="parent must precede"):
        load_observation_lineage(path)
