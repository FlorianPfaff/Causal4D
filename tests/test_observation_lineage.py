import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from causal4d.observation_lineage import (
    bind_twin_belief_observation_lineage,
    compute_observation_artifact_id,
    ObservationLineage,
    load_observation_lineage,
    validate_twin_belief_observation_lineage,
)

GOLDEN_ARTIFACT_ID = (
    "9c02e638f60424cca7738d347d1258acd208eb562f422efacd077db4edb2fe80"
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
    artifact_id: str = "c" * 64


def _write_observation(path: Path) -> None:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 1e-4
    factors = np.zeros((4, 3, 2))
    factors[:2, 0, 0] = 0.002
    factors[2:, 1, 1] = 0.003
    arrays = {
        "declared_frame_ids": np.asarray([8, 9], dtype=np.int64),
        "mean_xyz_m": np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.1, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "frame_ids": np.asarray([8, 8, 9, 9], dtype=np.int64),
        "entity_ids": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "view_indices": np.zeros(4, dtype=np.int64),
        "window_indices": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "correlation_group_ids": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "factor_group_ids": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "prior_reliability": np.asarray([0.9, 0.8, 0.7, 0.6]),
        "association_probability": np.ones(4),
        "local_covariance_m2": local,
        "low_rank_factor_m": factors,
        "group_ids": np.asarray([0, 1], dtype=np.int64),
        "group_prior_nominal_probability": np.asarray([0.85, 0.65]),
        "group_composite_weight": np.asarray([0.5, 0.5]),
    }
    descriptor = {
        "schema_name": "phys4d.observation_belief",
        "schema_version": 1,
        "case_id": "case-1",
        "stream_id": "prob4d:points",
        "causal_frame_stop": 12,
        "view_names": ["camera0"],
        "window_names": ["window0", "window1"],
        "factor_names": ["gauge_latent_0", "gauge_latent_1"],
        "source_repository": "FlorianPfaff/Prob4D",
        "source_revision": "a" * 40,
        "source_artifact_sha256": "b" * 64,
        "metadata": {"causal_source": "prefix only"},
    }
    artifact_id = compute_observation_artifact_id(descriptor, arrays)
    assert artifact_id == GOLDEN_ARTIFACT_ID
    descriptor["artifact_id"] = artifact_id
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        ),
        **arrays,
    )


def test_lineage_validates_golden_cross_repository_artifact(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    _write_observation(path)
    lineage = load_observation_lineage(path)

    assert lineage.artifact_id == GOLDEN_ARTIFACT_ID
    assert lineage.maximum_frame_id == 9
    assert lineage.factor_rank == 2


def test_binding_is_content_addressed_and_then_required(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    _write_observation(path)
    lineage = load_observation_lineage(path)
    twin = _TwinBelief(_Context("case-1", _Window(12)), {})

    with pytest.raises(ValueError, match="no source observation binding"):
        validate_twin_belief_observation_lineage(
            twin, lineage, require_bound=True
        )
    bound = bind_twin_belief_observation_lineage(twin, lineage)
    result = validate_twin_belief_observation_lineage(
        bound, lineage, require_bound=True
    )
    assert result["lineage_bound"]
    assert bound.metadata["source_observation_belief_id"] == GOLDEN_ARTIFACT_ID


def test_lineage_rejects_observation_beyond_twin_o_minus(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    _write_observation(path)
    lineage = load_observation_lineage(path)
    twin = _TwinBelief(_Context("case-1", _Window(10)), {})
    with pytest.raises(ValueError, match="beyond the TwinBelief O- boundary"):
        validate_twin_belief_observation_lineage(
            twin, lineage, require_bound=False
        )


def test_lineage_rejects_observation_before_twin_o_minus(tmp_path: Path) -> None:
    path = tmp_path / "observation.npz"
    _write_observation(path)
    lineage = load_observation_lineage(path)
    twin = _TwinBelief(_Context("case-1", _Window(12, frame_start=9)), {})
    with pytest.raises(ValueError, match="before the TwinBelief O- boundary"):
        validate_twin_belief_observation_lineage(
            twin, lineage, require_bound=False
        )


def test_lineage_provider_validation_is_deeply_immutable() -> None:
    validation = {"provider": {"checks": ["schema", {"passed": True}]}}
    lineage = ObservationLineage(
        artifact_id="a" * 64,
        case_id="case-1",
        stream_id="prob4d:points",
        causal_frame_stop=12,
        minimum_frame_id=8,
        maximum_frame_id=9,
        observation_count=4,
        group_count=2,
        factor_rank=2,
        source_repository="FlorianPfaff/Prob4D",
        source_revision="b" * 40,
        source_artifact_sha256="c" * 64,
        provider_validation=validation,
    )

    validation["provider"]["checks"][1]["passed"] = False
    assert lineage.provider_validation["provider"]["checks"][1]["passed"] is True
    with pytest.raises(TypeError, match="immutable"):
        lineage.provider_validation["provider"]["checks"].append("mutated")

    metadata = lineage.metadata()
    metadata["source_observation_provider_validation"]["provider"]["checks"].append(
        "copy-only"
    )
    assert "copy-only" not in lineage.provider_validation["provider"]["checks"]
