"""Method-neutral live acquisition-health decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

from causal4d.acquisition_flight_common import (
    HEALTH_SNAPSHOT_KIND,
    _assert_json_value,
    _parse_utc,
    _reject_target_outcomes,
    _require,
)

@dataclass(frozen=True)
class HealthThresholds:
    """Operational watchdog thresholds; not scientific acceptance thresholds."""

    maximum_heartbeat_age_s: float = 2.0
    maximum_clock_offset_ms: float = 5.0
    maximum_dropped_frames: int = 0
    minimum_free_bytes: int = 20 * 1024**3
    minimum_write_mib_s: float = 25.0

    def __post_init__(self) -> None:
        _require(
            self.maximum_heartbeat_age_s > 0.0,
            "heartbeat threshold must be positive",
        )
        _require(
            self.maximum_clock_offset_ms >= 0.0,
            "clock threshold must be nonnegative",
        )
        _require(
            isinstance(self.maximum_dropped_frames, int)
            and not isinstance(self.maximum_dropped_frames, bool)
            and self.maximum_dropped_frames >= 0,
            "dropped-frame threshold must be a nonnegative integer",
        )
        _require(self.minimum_free_bytes >= 0, "minimum_free_bytes must be nonnegative")
        _require(
            self.minimum_write_mib_s >= 0.0,
            "minimum_write_mib_s must be nonnegative",
        )


def _finite_number(value: Any, *, name: str, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{name} must be at least {minimum}")
    return result


def evaluate_health_snapshot(
    snapshot: Mapping[str, Any],
    *,
    thresholds: HealthThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate a live collection-health snapshot without reading target metrics."""

    settings = thresholds or HealthThresholds()
    values = dict(snapshot)
    _assert_json_value(values, name="health snapshot")
    _reject_target_outcomes(values)
    _require(values.get("schema_version") == 1, "unsupported health snapshot schema")
    _require(values.get("artifact_kind") == HEALTH_SNAPSHOT_KIND, "wrong snapshot kind")
    _require(
        values.get("target_outcomes_used") is False,
        "snapshot used target outcomes",
    )
    for field in ("protocol_id", "session_id"):
        _require(
            isinstance(values.get(field), str) and bool(values[field].strip()),
            f"snapshot {field} must be nonempty",
        )
    _parse_utc(values.get("captured_at_utc"), name="captured_at_utc")
    streams = values.get("streams")
    _require(
        isinstance(streams, Mapping) and bool(streams),
        "snapshot streams are missing",
    )
    failures: list[str] = []
    warnings: list[str] = []
    stream_results: dict[str, Any] = {}
    for stream_id, raw in sorted(streams.items(), key=lambda item: str(item[0])):
        _require(isinstance(raw, Mapping), f"stream {stream_id} must be an object")
        stream = dict(raw)
        required = stream.get("required", True)
        alive = stream.get("alive")
        _require(
            isinstance(required, bool),
            f"stream {stream_id}.required must be boolean",
        )
        _require(isinstance(alive, bool), f"stream {stream_id}.alive must be boolean")
        heartbeat = _finite_number(
            stream.get("heartbeat_age_s"),
            name=f"stream {stream_id}.heartbeat_age_s",
            minimum=0.0,
        )
        dropped = stream.get("dropped_frames", 0)
        _require(
            isinstance(dropped, int) and not isinstance(dropped, bool) and dropped >= 0,
            f"stream {stream_id}.dropped_frames must be nonnegative integer",
        )
        clock_offset = stream.get("clock_offset_ms")
        if clock_offset is not None:
            clock_offset = _finite_number(
                clock_offset,
                name=f"stream {stream_id}.clock_offset_ms",
            )
        issues = []
        if required and not alive:
            issues.append("required_stream_not_alive")
        if required and heartbeat > settings.maximum_heartbeat_age_s:
            issues.append("required_stream_heartbeat_stale")
        if required and dropped > settings.maximum_dropped_frames:
            issues.append("dropped_frame_limit_exceeded")
        if (
            required
            and clock_offset is not None
            and abs(clock_offset) > settings.maximum_clock_offset_ms
        ):
            issues.append("clock_offset_limit_exceeded")
        if issues:
            failures.extend(f"stream:{stream_id}:{issue}" for issue in issues)
        elif not required and (
            not alive or heartbeat > settings.maximum_heartbeat_age_s
        ):
            warnings.append(f"optional_stream:{stream_id}:unhealthy")
        stream_results[str(stream_id)] = {
            "required": required,
            "alive": alive,
            "heartbeat_age_s": heartbeat,
            "dropped_frames": dropped,
            "clock_offset_ms": clock_offset,
            "issues": issues,
        }

    storage = values.get("storage")
    _require(isinstance(storage, Mapping), "snapshot storage is missing")
    free_bytes = storage.get("free_bytes")
    _require(
        isinstance(free_bytes, int)
        and not isinstance(free_bytes, bool)
        and free_bytes >= 0,
        "storage.free_bytes must be a nonnegative integer",
    )
    write_rate = _finite_number(
        storage.get("write_mib_s"),
        name="storage.write_mib_s",
        minimum=0.0,
    )
    if free_bytes < settings.minimum_free_bytes:
        failures.append("storage:free_space_below_threshold")
    if write_rate < settings.minimum_write_mib_s:
        failures.append("storage:write_rate_below_threshold")
    passed = not failures
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DAcquisitionHealthDecision",
        "protocol_id": values["protocol_id"],
        "session_id": values["session_id"],
        "execution_id": values.get("execution_id"),
        "captured_at_utc": values["captured_at_utc"],
        "thresholds": asdict(settings),
        "streams": stream_results,
        "storage": {"free_bytes": free_bytes, "write_mib_s": write_rate},
        "failures": failures,
        "warnings": warnings,
        "passed": passed,
        "target_outcomes_used": False,
    }



__all__ = [
    "HEALTH_SNAPSHOT_KIND",
    "HealthThresholds",
    "evaluate_health_snapshot",
]
