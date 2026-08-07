from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest

from causal4d.baselines import PredictiveDistribution
from causal4d.hybrid_reliability import (
    HybridReliabilityCase,
    fit_hybrid_reliability_calibration,
    load_hybrid_reliability_calibration,
    save_hybrid_reliability_calibration,
)
from causal4d.hybrid_reliability_admission import (
    load_claim_bearing_hybrid_reliability_calibration,
)


def _prediction(method: str, value: float) -> PredictiveDistribution:
    mean = np.full((6, 1, 2), value, dtype=float)
    return PredictiveDistribution(
        method=method,
        mean=mean,
        variance=np.ones_like(mean),
    )


def _case(case_id: str, physics: float, hybrid: float) -> HybridReliabilityCase:
    return HybridReliabilityCase(
        case_id=case_id,
        physics=_prediction("physics_only", physics),
        hybrid=_prediction("hybrid", hybrid),
        observations=np.zeros((6, 1, 2), dtype=float),
        descriptor_leverage=0.2,
        prefix_frame_count=3,
    )


def _calibration():
    return fit_hybrid_reliability_calibration(
        (
            _case("source-b", 0.24, 0.12),
            _case("source-a", 0.20, 0.10),
        )
    )


def _readdress(payload: dict[str, Any]) -> None:
    canonical = dict(payload)
    canonical.pop("calibration_id", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["calibration_id"] = hashlib.sha256(encoded).hexdigest()


def test_claim_bearing_loader_accepts_exact_independent_identity(tmp_path) -> None:
    calibration = _calibration()
    path = tmp_path / "calibration.json"
    save_hybrid_reliability_calibration(path, calibration)

    loaded = load_claim_bearing_hybrid_reliability_calibration(
        path,
        expected_calibration_id=calibration.calibration_id,
    )

    assert loaded == calibration


def test_fully_readdressed_policy_change_requires_new_independent_identity(
    tmp_path,
) -> None:
    original = _calibration()
    payload = original.as_dict()
    payload["minimum_mean_source_future_relative_improvement"] = 0.95
    payload["hybrid_enabled"] = False
    _readdress(payload)
    path = tmp_path / "readdressed-calibration.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    internally_valid = load_hybrid_reliability_calibration(path)
    assert internally_valid.calibration_id == payload["calibration_id"]
    assert internally_valid.calibration_id != original.calibration_id

    with pytest.raises(ValueError, match="independently frozen identity"):
        load_claim_bearing_hybrid_reliability_calibration(
            path,
            expected_calibration_id=original.calibration_id,
        )


def test_claim_bearing_loader_rejects_invalid_expected_identity(tmp_path) -> None:
    calibration = _calibration()
    path = tmp_path / "calibration.json"
    save_hybrid_reliability_calibration(path, calibration)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_claim_bearing_hybrid_reliability_calibration(
            path,
            expected_calibration_id="not-a-digest",
        )
