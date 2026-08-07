from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from causal4d.contracts import array_sha256
from causal4d.prob4d_joint_observation import joint_observation_from_prob4d


def _descriptor() -> dict[str, object]:
    return {
        "case_id": "provenance-case",
        "source_revision": "a" * 40,
        "source_artifact_sha256": "b" * 64,
    }


def _arrays() -> dict[str, np.ndarray]:
    return {
        "mean_xyz_m": np.array([[0.1, 0.2, 0.3]]),
        "local_covariance_m2": np.array([np.eye(3) * 0.01]),
        "low_rank_factor_m": np.array([[[0.02], [0.0], [0.01]]]),
        "frame_ids": np.array([2], dtype=np.int64),
        "entity_ids": np.array([4], dtype=np.int64),
        "factor_group_ids": np.array([0], dtype=np.int64),
        "association_probability": np.ones(1),
        "prior_reliability": np.ones(1),
        "group_prior_nominal_probability": np.ones(1),
        "group_composite_weight": np.ones(1),
    }


def test_provider_validation_is_recursively_immutable(monkeypatch) -> None:
    source_validation = {"validated": True, "nested": {"count": 1}}
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: source_validation,
    )

    evidence, diagnostics = joint_observation_from_prob4d(
        _descriptor(),
        _arrays(),
        rollout_frame_ids=(1, 2),
        entity_to_node={4: 0},
    )

    assert isinstance(diagnostics.provider_validation, MappingProxyType)
    assert isinstance(diagnostics.provider_validation["nested"], MappingProxyType)
    with pytest.raises(TypeError):
        diagnostics.provider_validation["validated"] = False
    with pytest.raises(TypeError):
        diagnostics.provider_validation["nested"]["count"] = 2
    source_validation["nested"]["count"] = 99
    assert diagnostics.provider_validation["nested"]["count"] == 1
    assert evidence.metadata["provider_validation"]["nested"]["count"] == 1


def test_adapter_binds_exact_source_array_bytes_and_full_digest(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    descriptor = _descriptor()
    arrays = _arrays()
    arrays["association_probability"] = np.ones(1, dtype=np.float32)
    arrays["prior_reliability"] = np.ones(1, dtype=np.float32)

    evidence, _ = joint_observation_from_prob4d(
        descriptor,
        arrays,
        rollout_frame_ids=(1, 2),
        entity_to_node={4: 0},
    )

    assert evidence.evidence_id.endswith(str(descriptor["source_artifact_sha256"]))
    assert evidence.metadata["association_probability_sha256"] == array_sha256(
        arrays["association_probability"]
    )
    assert evidence.metadata["prior_reliability_sha256"] == array_sha256(
        arrays["prior_reliability"]
    )
    assert evidence.metadata["association_probability_sha256"] != array_sha256(
        arrays["association_probability"].astype(float)
    )


def test_adapter_rechecks_source_identifiers_after_validator(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    invalid_revision = _descriptor()
    invalid_revision["source_revision"] = "not-a-commit"
    with pytest.raises(ValueError, match="source_revision"):
        joint_observation_from_prob4d(
            invalid_revision,
            _arrays(),
            rollout_frame_ids=(1, 2),
            entity_to_node={4: 0},
        )

    invalid_digest = _descriptor()
    invalid_digest["source_artifact_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="source_artifact_sha256"):
        joint_observation_from_prob4d(
            invalid_digest,
            _arrays(),
            rollout_frame_ids=(1, 2),
            entity_to_node={4: 0},
        )


def test_adapter_rejects_non_json_provider_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True, "bad": object()},
    )

    with pytest.raises(ValueError, match="provider validation"):
        joint_observation_from_prob4d(
            _descriptor(),
            _arrays(),
            rollout_frame_ids=(1, 2),
            entity_to_node={4: 0},
        )
