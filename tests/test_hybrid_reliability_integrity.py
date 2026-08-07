from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest

from causal4d.baselines import PredictiveDistribution
from causal4d.hybrid_reliability import (
    HybridReliabilityCalibration,
    HybridReliabilityCase,
    fit_hybrid_reliability_calibration,
    load_hybrid_reliability_calibration,
    save_hybrid_reliability_calibration,
)


def _prediction(
    method: str,
    value: float | np.ndarray,
    *,
    variance: float = 1.0,
) -> PredictiveDistribution:
    mean = (
        np.full((6, 1, 2), float(value), dtype=float)
        if np.isscalar(value)
        else np.asarray(value, dtype=float)
    )
    return PredictiveDistribution(
        method=method,
        mean=mean,
        variance=np.full_like(mean, variance, dtype=float),
    )


def _case(
    case_id: str,
    *,
    physics: float = 0.20,
    hybrid: float = 0.10,
    leverage: float = 0.20,
) -> HybridReliabilityCase:
    return HybridReliabilityCase(
        case_id=case_id,
        physics=_prediction("physics_only", physics),
        hybrid=_prediction("hybrid", hybrid),
        observations=np.zeros((6, 1, 2), dtype=float),
        descriptor_leverage=leverage,
        prefix_frame_count=3,
    )


def _sources() -> tuple[HybridReliabilityCase, HybridReliabilityCase]:
    return (
        _case("source-b", physics=0.24, hybrid=0.12, leverage=0.40),
        _case("source-a", physics=0.20, hybrid=0.10, leverage=0.20),
    )


def _calibration() -> HybridReliabilityCalibration:
    return fit_hybrid_reliability_calibration(
        _sources(),
        prefix_rmse_margin=0.01,
        prefix_log_score_margin=0.02,
        support_margin=1.25,
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


def _write_payload(tmp_path, payload: dict[str, Any]):
    path = tmp_path / "hybrid-reliability.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_source_order_does_not_change_calibration_identity() -> None:
    sources = _sources()
    forward = fit_hybrid_reliability_calibration(
        sources,
        prefix_rmse_margin=0.01,
        prefix_log_score_margin=0.02,
        support_margin=1.25,
    )
    reverse = fit_hybrid_reliability_calibration(
        tuple(reversed(sources)),
        prefix_rmse_margin=0.01,
        prefix_log_score_margin=0.02,
        support_margin=1.25,
    )

    assert forward.source_case_ids == ("source-a", "source-b")
    assert forward.as_dict() == reverse.as_dict()
    assert forward.calibration_id == reverse.calibration_id


def test_noncanonical_direct_source_order_is_rejected() -> None:
    calibration = _calibration()
    payload = calibration.as_dict()
    payload.pop("schema_version")
    payload.pop("artifact_kind")
    payload.pop("calibration_id")
    for field in (
        "source_case_ids",
        "source_case_artifact_ids",
        "source_prefix_input_ids",
        "source_prefix_rmse_relative_improvements",
        "source_prefix_log_score_gains",
        "source_correction_standard_deviation_ratios",
        "source_descriptor_leverages",
        "source_future_relative_improvements",
    ):
        payload[field] = tuple(reversed(payload[field]))

    with pytest.raises(ValueError, match="deterministic sorted order"):
        HybridReliabilityCalibration(**payload)


@pytest.mark.parametrize(
    "field,replacement,match",
    [
        (
            "minimum_prefix_rmse_relative_improvement",
            0.123,
            "minimum_prefix_rmse_relative_improvement does not match",
        ),
        (
            "minimum_prefix_log_score_gain",
            0.456,
            "minimum_prefix_log_score_gain does not match",
        ),
        (
            "maximum_correction_standard_deviation_ratio",
            9.0,
            "maximum_correction_standard_deviation_ratio does not match",
        ),
        (
            "maximum_descriptor_leverage",
            9.0,
            "maximum_descriptor_leverage does not match",
        ),
        (
            "mean_source_future_relative_improvement",
            0.123,
            "mean_source_future_relative_improvement does not match",
        ),
        (
            "source_future_win_fraction",
            0.25,
            "source_future_win_fraction does not match",
        ),
    ],
)
def test_readdressed_derived_field_tampering_fails_closed(
    tmp_path,
    field: str,
    replacement: float,
    match: str,
) -> None:
    payload = _calibration().as_dict()
    payload[field] = replacement
    _readdress(payload)
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match=match):
        load_hybrid_reliability_calibration(path)


def test_readdressed_enabled_flag_tampering_fails_closed(tmp_path) -> None:
    payload = _calibration().as_dict()
    payload["hybrid_enabled"] = not payload["hybrid_enabled"]
    _readdress(payload)
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="recomputed source gates"):
        load_hybrid_reliability_calibration(path)


def test_one_source_artifact_is_rejected_even_when_derivations_are_consistent(
    tmp_path,
) -> None:
    payload = _calibration().as_dict()
    vector_fields = (
        "source_case_ids",
        "source_case_artifact_ids",
        "source_prefix_input_ids",
        "source_prefix_rmse_relative_improvements",
        "source_prefix_log_score_gains",
        "source_correction_standard_deviation_ratios",
        "source_descriptor_leverages",
        "source_future_relative_improvements",
    )
    for field in vector_fields:
        payload[field] = payload[field][:1]

    rmse = payload["source_prefix_rmse_relative_improvements"][0]
    log_score = payload["source_prefix_log_score_gains"][0]
    correction = payload["source_correction_standard_deviation_ratios"][0]
    leverage = payload["source_descriptor_leverages"][0]
    future = payload["source_future_relative_improvements"][0]
    payload["minimum_prefix_rmse_relative_improvement"] = max(
        0.0,
        rmse - payload["prefix_rmse_margin"],
    )
    payload["minimum_prefix_log_score_gain"] = max(
        0.0,
        log_score - payload["prefix_log_score_margin"],
    )
    payload["maximum_correction_standard_deviation_ratio"] = max(
        correction * payload["support_margin"],
        1e-12,
    )
    payload["maximum_descriptor_leverage"] = max(
        leverage * payload["support_margin"],
        1e-12,
    )
    payload["mean_source_future_relative_improvement"] = future
    payload["source_future_win_fraction"] = float(future > 0.0)
    payload["hybrid_enabled"] = bool(
        future >= payload["minimum_mean_source_future_relative_improvement"]
        and payload["source_future_win_fraction"]
        >= payload["minimum_source_future_win_fraction"]
    )
    _readdress(payload)
    path = _write_payload(tmp_path, payload)

    with pytest.raises(ValueError, match="at least two unique cases"):
        load_hybrid_reliability_calibration(path)


def test_nonzero_derivation_margins_round_trip(tmp_path) -> None:
    calibration = _calibration()
    path = tmp_path / "hybrid-reliability.json"

    save_hybrid_reliability_calibration(path, calibration)
    loaded = load_hybrid_reliability_calibration(path)

    assert loaded == calibration
    assert loaded.prefix_rmse_margin == pytest.approx(0.01)
    assert loaded.prefix_log_score_margin == pytest.approx(0.02)
    assert loaded.support_margin == pytest.approx(1.25)
