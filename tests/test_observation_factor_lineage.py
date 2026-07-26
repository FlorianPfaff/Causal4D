import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from causal4d.observation_factor_lineage import (
    OBSERVATION_FACTOR_SCHEMA,
    bind_twin_belief_observation_factor_lineage,
    compute_observation_factor_bundle_id,
    file_sha256,
    load_observation_factor_lineage,
    validate_twin_belief_observation_factor_lineage,
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(
    root: Path,
    *,
    case_id: str = "case-1",
    frame_index: int = 8,
    reliability: np.ndarray | None = None,
    association: np.ndarray | None = None,
    composite_weight: float = 0.5,
    extra_array: bool = False,
) -> Path:
    payload = root / "factors.npz"
    arrays = {
        "gauge_mean": np.zeros(7),
        "gauge_covariance": np.eye(7) * 1e-4,
        "point_ids": np.asarray([10, 11], dtype=np.int64),
        "points_local_m": np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "valid_mask": np.asarray([True, True]),
        "local_covariance_m2": np.tile(np.eye(3) * 1e-4, (2, 1, 1)),
        "association_probability": (
            np.asarray([0.9, 0.8])
            if association is None
            else np.asarray(association)
        ),
        "prior_reliability": (
            np.asarray([0.7, 0.6])
            if reliability is None
            else np.asarray(reliability)
        ),
    }
    if extra_array:
        arrays["unexpected"] = np.asarray([1])
    np.savez_compressed(payload, **arrays)
    record = {
        "schema": OBSERVATION_FACTOR_SCHEMA,
        "schema_version": 3,
        "gauge_parameterization": "log-scale-rotvec-translation-v1",
        "sequence_id": "sequence-1",
        "case_id": case_id,
        "stream_id": "prob4d:camera0",
        "source_repository": "FlorianPfaff/Prob4D",
        "source_revision": "a" * 40,
        "causal_frame_stop": 12,
        "causal_frame_stop_convention": "exclusive",
        "metadata": {},
        "payload": {
            "path": payload.name,
            "sha256": _sha(payload),
            "allow_pickle": False,
        },
        "gauges": [
            {
                "gauge_id": "window-0",
                "mean_key": "gauge_mean",
                "covariance_key": "gauge_covariance",
            }
        ],
        "factors": [
            {
                "factor_id": "factor-0",
                "frame_index": frame_index,
                "view_id": "camera0",
                "window_id": "window-0",
                "gauge_id": "window-0",
                "correlation_group_id": "shared-frame-8",
                "causal_frame_stop": 12,
                "prior_nominal_probability": 0.8,
                "composite_weight": composite_weight,
                "arrays": {
                    "point_ids": "point_ids",
                    "points_local_m": "points_local_m",
                    "valid_mask": "valid_mask",
                    "local_covariance_m2": "local_covariance_m2",
                    "association_probability": "association_probability",
                    "prior_reliability": "prior_reliability",
                },
                "ray_directions_local_key": None,
            }
        ],
    }
    manifest = root / "factors.json"
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_loads_exact_factor_bundle_and_computes_pair_identity(
    tmp_path: Path,
) -> None:
    manifest = _write_bundle(tmp_path)
    lineage = load_observation_factor_lineage(manifest)

    assert lineage.case_id == "case-1"
    assert lineage.stream_id == "prob4d:camera0"
    assert lineage.minimum_frame_id == 8
    assert lineage.maximum_frame_id == 8
    assert lineage.observation_count == 2
    assert lineage.active_observation_count == 2
    assert lineage.artifact_id == compute_observation_factor_bundle_id(
        file_sha256(manifest),
        lineage.payload_sha256,
    )


def test_binding_requires_exact_metadata_and_preserves_content_address(
    tmp_path: Path,
) -> None:
    lineage = load_observation_factor_lineage(_write_bundle(tmp_path))
    twin = _TwinBelief(_Context("case-1", _Window(12)), {})

    with pytest.raises(ValueError, match="no source factor-bundle binding"):
        validate_twin_belief_observation_factor_lineage(
            twin,
            lineage,
            require_bound=True,
        )
    bound = bind_twin_belief_observation_factor_lineage(twin, lineage)
    result = validate_twin_belief_observation_factor_lineage(
        bound,
        lineage,
        require_bound=True,
    )
    assert result["lineage_bound"]
    assert (
        bound.metadata["source_observation_factor_bundle_id"]
        == lineage.artifact_id
    )

    changed = dict(bound.metadata)
    changed["source_observation_factor_revision"] = "wrong"
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_twin_belief_observation_factor_lineage(
            _TwinBelief(bound.context, changed),
            lineage,
            require_bound=True,
        )


def test_rejects_payload_tampering(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    payload = tmp_path / "factors.npz"
    payload.write_bytes(payload.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_observation_factor_lineage(manifest)


def test_rejects_case_and_causal_boundary_mismatch(tmp_path: Path) -> None:
    lineage = load_observation_factor_lineage(_write_bundle(tmp_path))

    with pytest.raises(ValueError, match="different cases"):
        validate_twin_belief_observation_factor_lineage(
            _TwinBelief(_Context("other", _Window(12)), {}),
            lineage,
            require_bound=False,
        )
    with pytest.raises(ValueError, match="beyond"):
        validate_twin_belief_observation_factor_lineage(
            _TwinBelief(_Context("case-1", _Window(10)), {}),
            lineage,
            require_bound=False,
        )
    with pytest.raises(ValueError, match="before"):
        validate_twin_belief_observation_factor_lineage(
            _TwinBelief(_Context("case-1", _Window(12, frame_start=9)), {}),
            lineage,
            require_bound=False,
        )


def test_rejects_invalid_reliability_even_when_association_is_valid(
    tmp_path: Path,
) -> None:
    manifest = _write_bundle(
        tmp_path,
        reliability=np.asarray([1.2, 0.5]),
        association=np.asarray([0.9, 0.8]),
    )

    with pytest.raises(ValueError, match="prior reliability"):
        load_observation_factor_lineage(manifest)


def test_rejects_unexpected_payload_array(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path, extra_array=True)

    with pytest.raises(ValueError, match="payload array set changed"):
        load_observation_factor_lineage(manifest)


def test_rejects_nonexclusive_cutoff(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["causal_frame_stop_convention"] = "inclusive"
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="must be exclusive"):
        load_observation_factor_lineage(manifest)


def test_rejects_invalid_composite_weight(tmp_path: Path) -> None:
    manifest = _write_bundle(tmp_path, composite_weight=0.0)

    with pytest.raises(ValueError, match="composite_weight"):
        load_observation_factor_lineage(manifest)
