from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from causal4d.observation_clock_offset_prior import (
    OBSERVATION_TIME_CORRECTION_CONVENTION,
    ObservationClockOffsetPriorV1,
    fit_observation_clock_offset_prior,
    load_observation_clock_offset_prior,
    write_observation_clock_offset_prior,
)


def _artifact_id(value: dict[str, Any]) -> str:
    canonical = dict(value)
    canonical.pop("artifact_id", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibration(
    execution_id: str,
    offset_s: float,
    *,
    target_outcomes_used: bool = False,
    convention: str = "aligned_measurement_time_s = measurement_time_s + offset_s",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "ActuatorRealizationCalibration",
        "execution_id": execution_id,
        "timestamp_alignment": {
            "convention": convention,
            "best_offset_s": offset_s,
            "offset_grid_s": [
                -0.020,
                -0.019,
                -0.018,
                -0.017,
                -0.016,
                -0.015,
                -0.014,
                -0.013,
                -0.012,
                -0.011,
                -0.010,
                -0.009,
                -0.008,
                -0.007,
                -0.006,
                -0.005,
                -0.004,
                -0.003,
                -0.002,
                -0.001,
                0.000,
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.010,
                0.011,
                0.012,
                0.013,
                0.014,
                0.015,
                0.016,
                0.017,
                0.018,
                0.019,
                0.020,
            ],
        },
        "information_boundary": {
            "source_or_dry_run_only": True,
            "target_outcomes_used": target_outcomes_used,
            "hardware_timestamps_authoritative": True,
            "bias_model_is_not_a_physical_frame_or_gain_posterior": True,
        },
        "claim_boundary": "source-only timing diagnostic",
    }
    value["artifact_id"] = _artifact_id(value)
    return value


def _fit(
    calibrations: list[dict[str, Any]] | None = None,
) -> ObservationClockOffsetPriorV1:
    source = calibrations or [
        _calibration("source-03", -0.009),
        _calibration("source-01", -0.011),
        _calibration("source-02", -0.010),
    ]
    return fit_observation_clock_offset_prior(
        source,
        clock_domain="camera-hardware-clock",
        reference_clock_domain="actuator-command-clock",
        time_scale="device-monotonic",
        source_revision="a" * 40,
        minimum_predictive_standard_deviation_s=5e-4,
    )


def test_fit_uses_equal_execution_predictive_variance() -> None:
    prior = _fit()

    assert prior.execution_ids == ("source-01", "source-02", "source-03")
    assert prior.source_offsets_s == pytest.approx((-0.011, -0.010, -0.009))
    assert prior.mean_offset_s == pytest.approx(-0.010)
    assert prior.sample_standard_deviation_s == pytest.approx(0.001)
    quantization = 0.001 / math.sqrt(12.0)
    expected = math.sqrt((1.0 + 1.0 / 3.0) * 0.001**2 + quantization**2)
    assert prior.grid_quantization_standard_deviation_s == pytest.approx(quantization)
    assert prior.predictive_standard_deviation_s == pytest.approx(expected)
    assert prior.source_group_count == 3


def test_input_order_does_not_change_content_identity() -> None:
    calibrations = [
        _calibration("source-01", -0.011),
        _calibration("source-02", -0.010),
        _calibration("source-03", -0.009),
    ]

    forward = _fit(calibrations)
    reverse = _fit(list(reversed(calibrations)))

    assert forward.artifact_id == reverse.artifact_id
    assert forward.to_record() == reverse.to_record()


def test_floor_prevents_zero_width_prior() -> None:
    prior = _fit(
        [
            _calibration("source-01", -0.010),
            _calibration("source-02", -0.010),
            _calibration("source-03", -0.010),
        ]
    )

    assert prior.sample_standard_deviation_s == 0.0
    assert prior.predictive_standard_deviation_s == pytest.approx(5e-4)


def test_bayesian_phystwin_payload_preserves_sign_convention() -> None:
    prior = _fit()

    assert prior.bayesian_phystwin_prior_payload() == {
        "clock_domain": "camera-hardware-clock",
        "mean_offset_s": pytest.approx(-0.010),
        "standard_deviation_s": pytest.approx(
            prior.predictive_standard_deviation_s
        ),
        "source_artifact_id": prior.artifact_id,
        "offset_convention": OBSERVATION_TIME_CORRECTION_CONVENTION,
    }


def test_tampered_source_calibration_is_rejected() -> None:
    tampered = _calibration("source-01", -0.011)
    tampered["timestamp_alignment"]["best_offset_s"] = -0.012

    with pytest.raises(ValueError, match="artifact ID mismatch"):
        _fit(
            [
                tampered,
                _calibration("source-02", -0.010),
                _calibration("source-03", -0.009),
            ]
        )


def test_target_outcomes_and_changed_convention_are_rejected() -> None:
    with pytest.raises(ValueError, match="source-only information boundary"):
        _fit(
            [
                _calibration("source-01", -0.011, target_outcomes_used=True),
                _calibration("source-02", -0.010),
                _calibration("source-03", -0.009),
            ]
        )

    with pytest.raises(ValueError, match="timing convention changed"):
        _fit(
            [
                _calibration(
                    "source-01",
                    -0.011,
                    convention="observation_time_s = reference_time_s + offset_s",
                ),
                _calibration("source-02", -0.010),
                _calibration("source-03", -0.009),
            ]
        )


def test_too_few_or_duplicate_source_groups_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least three"):
        _fit(
            [
                _calibration("source-01", -0.011),
                _calibration("source-02", -0.010),
            ]
        )

    with pytest.raises(ValueError, match="execution IDs must be unique"):
        _fit(
            [
                _calibration("source-01", -0.011),
                _calibration("source-01", -0.010),
                _calibration("source-03", -0.009),
            ]
        )


def test_derived_summary_cannot_be_forged() -> None:
    prior = _fit()
    record = prior.to_record()
    record["predictive_standard_deviation_s"] *= 0.5
    record["artifact_id"] = _artifact_id(record)

    with pytest.raises(ValueError, match="summary does not match"):
        ObservationClockOffsetPriorV1.from_record(record)


def test_round_trip_and_idempotent_no_clobber_publication(tmp_path: Path) -> None:
    prior = _fit()
    path = tmp_path / "clock-offset-prior.json"

    write_observation_clock_offset_prior(prior, path)
    write_observation_clock_offset_prior(prior, path)
    loaded = load_observation_clock_offset_prior(path)

    assert loaded.artifact_id == prior.artifact_id
    assert loaded.to_record() == prior.to_record()

    different = fit_observation_clock_offset_prior(
        [
            _calibration("source-01", -0.012),
            _calibration("source-02", -0.010),
            _calibration("source-03", -0.008),
        ],
        clock_domain="camera-hardware-clock",
        reference_clock_domain="actuator-command-clock",
        time_scale="device-monotonic",
        source_revision="a" * 40,
    )
    with pytest.raises(ValueError, match="different content"):
        write_observation_clock_offset_prior(different, path)


def test_loader_rejects_duplicate_keys_tampering_and_symlinks(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"first","schema":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        load_observation_clock_offset_prior(duplicate)

    prior = _fit()
    tampered = prior.to_record()
    tampered["mean_offset_s"] = -0.020
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="summary does not match"):
        load_observation_clock_offset_prior(path)

    target = tmp_path / "target.json"
    write_observation_clock_offset_prior(prior, target)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        load_observation_clock_offset_prior(symlink)
