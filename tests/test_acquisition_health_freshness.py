from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal4d.acquisition_health import (
    HEALTH_SNAPSHOT_KIND,
    HealthThresholds,
    evaluate_health_snapshot,
    evaluate_health_snapshot_file,
)


CAPTURED = "2026-08-03T08:00:00+00:00"


def _snapshot() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": HEALTH_SNAPSHOT_KIND,
        "protocol_id": "protocol-v1",
        "session_id": "session-1",
        "execution_id": "execution-1",
        "captured_at_utc": CAPTURED,
        "target_outcomes_used": False,
        "streams": {
            "rgbd": {
                "required": True,
                "alive": True,
                "heartbeat_age_s": 0.2,
                "dropped_frames": 0,
                "clock_offset_ms": 1.5,
            }
        },
        "storage": {"free_bytes": 1000, "write_mib_s": 100.0},
    }


def _thresholds(**changes: object) -> HealthThresholds:
    values: dict[str, object] = {
        "maximum_heartbeat_age_s": 2.0,
        "maximum_clock_offset_ms": 5.0,
        "maximum_dropped_frames": 0,
        "minimum_free_bytes": 500,
        "minimum_write_mib_s": 50.0,
        "maximum_snapshot_age_s": 5.0,
        "maximum_future_skew_s": 1.0,
    }
    values.update(changes)
    return HealthThresholds(**values)


def test_direct_mapping_evaluation_remains_deterministic_offline() -> None:
    result = evaluate_health_snapshot(_snapshot(), thresholds=_thresholds())

    assert result["passed"] is True
    assert result["snapshot_age_s"] == 0.0
    assert result["evaluated_at_utc"] == CAPTURED
    assert result["streams"]["rgbd"]["effective_heartbeat_age_s"] == 0.2


def test_transport_delay_contributes_to_effective_heartbeat_age() -> None:
    result = evaluate_health_snapshot(
        _snapshot(),
        thresholds=_thresholds(maximum_snapshot_age_s=10.0),
        evaluated_at_utc="2026-08-03T08:00:03+00:00",
    )

    assert result["snapshot_age_s"] == 3.0
    assert result["streams"]["rgbd"]["effective_heartbeat_age_s"] == 3.2
    assert "stream:rgbd:required_stream_heartbeat_stale" in result["failures"]


def test_snapshot_age_and_future_skew_fail_closed() -> None:
    stale = evaluate_health_snapshot(
        _snapshot(),
        thresholds=_thresholds(maximum_snapshot_age_s=2.0),
        evaluated_at_utc="2026-08-03T08:00:03+00:00",
    )
    assert "snapshot:age_limit_exceeded" in stale["failures"]

    future = evaluate_health_snapshot(
        _snapshot(),
        thresholds=_thresholds(maximum_future_skew_s=0.5),
        evaluated_at_utc="2026-08-03T07:59:58+00:00",
    )
    assert "snapshot:captured_in_future" in future["failures"]


def test_file_evaluation_binds_exact_bytes_and_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "health.json"
    path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    result = evaluate_health_snapshot_file(
        path,
        thresholds=_thresholds(),
        evaluated_at_utc=CAPTURED,
    )

    assert len(result["snapshot_sha256"]) == 64
    assert result["snapshot_byte_count"] == path.stat().st_size
    assert len(result["decision_sha256"]) == 64

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        evaluate_health_snapshot_file(duplicate, evaluated_at_utc=CAPTURED)


def test_file_evaluation_rejects_symbolic_links(tmp_path: Path) -> None:
    target = tmp_path / "health.json"
    target.write_text(json.dumps(_snapshot()), encoding="utf-8")
    link = tmp_path / "health-link.json"
    try:
        link.symlink_to(target)
    except OSError as error:  # pragma: no cover - symlinks unavailable
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="ordinary readable file"):
        evaluate_health_snapshot_file(link, evaluated_at_utc=CAPTURED)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("maximum_heartbeat_age_s", float("inf")),
        ("maximum_clock_offset_ms", float("nan")),
        ("minimum_free_bytes", True),
        ("maximum_dropped_frames", False),
    ),
)
def test_thresholds_reject_nonfinite_and_boolean_values(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        _thresholds(**{name: value})
