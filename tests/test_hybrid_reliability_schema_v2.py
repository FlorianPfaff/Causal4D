from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from causal4d.baselines import PredictiveDistribution
from causal4d.hybrid_reliability import (
    HybridReliabilityCase,
    fit_hybrid_reliability_calibration,
    load_hybrid_reliability_calibration,
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


def test_schema_version_one_is_not_silently_upgraded(tmp_path) -> None:
    calibration = fit_hybrid_reliability_calibration(
        (
            _case("source-b", 0.24, 0.12),
            _case("source-a", 0.20, 0.10),
        )
    )
    payload = calibration.as_dict()
    payload["schema_version"] = 1
    canonical = dict(payload)
    canonical.pop("calibration_id")
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["calibration_id"] = hashlib.sha256(encoded).hexdigest()
    path = tmp_path / "legacy-calibration.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported.*schema"):
        load_hybrid_reliability_calibration(path)
