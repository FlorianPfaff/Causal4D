"""Schema and state machine for the acquisition flight-recorder journal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any, Literal

from causal4d.acquisition_flight_common import (
    JOURNAL_EVENT_KIND,
    JOURNAL_SCHEMA_VERSION,
    _assert_json_value,
    _canonical_sha256,
    _is_sha256,
    _parse_utc,
    _reject_target_outcomes,
    _require,
    _utc_now,
)

JournalOutcome = Literal["completed", "aborted"]

_ALLOWED_EVENT_TYPES = frozenset(
    {
        "session_started",
        "execution_started",
        "stream_heartbeat",
        "clock_offset_sample",
        "storage_sample",
        "technical_warning",
        "technical_failure",
        "operator_note",
        "artifact_closed",
        "execution_completed",
        "execution_aborted",
        "recovery_started",
        "recovery_completed",
        "session_completed",
        "session_aborted",
    }
)
_FINAL_EVENT_TYPES: dict[str, JournalOutcome] = {
    "session_completed": "completed",
    "session_aborted": "aborted",
}


def _validate_event_shape(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    _require(
        payload.get("schema_version") == JOURNAL_SCHEMA_VERSION,
        "unsupported journal event schema",
    )
    _require(payload.get("artifact_kind") == JOURNAL_EVENT_KIND, "wrong event kind")
    for field in ("protocol_id", "session_id", "event_type", "source"):
        _require(
            isinstance(payload.get(field), str) and bool(payload[field].strip()),
            f"journal event {field} must be nonempty",
        )
    _require(
        payload["event_type"] in _ALLOWED_EVENT_TYPES,
        f"unsupported journal event type: {payload['event_type']}",
    )
    execution_id = payload.get("execution_id")
    _require(
        execution_id is None
        or (isinstance(execution_id, str) and bool(execution_id.strip())),
        "execution_id must be null or nonempty",
    )
    sequence = payload.get("sequence")
    _require(
        isinstance(sequence, int) and not isinstance(sequence, bool) and sequence >= 0,
        "journal sequence must be a nonnegative integer",
    )
    monotonic_ns = payload.get("monotonic_ns")
    _require(
        isinstance(monotonic_ns, int)
        and not isinstance(monotonic_ns, bool)
        and monotonic_ns >= 0,
        "journal monotonic_ns must be a nonnegative integer",
    )
    _parse_utc(payload.get("recorded_at_utc"), name="recorded_at_utc")
    previous = payload.get("previous_event_sha256")
    _require(
        previous is None or _is_sha256(previous),
        "previous_event_sha256 must be null or lowercase SHA-256",
    )
    _require(
        isinstance(payload.get("payload"), Mapping),
        "event payload must be an object",
    )
    _assert_json_value(payload["payload"], name="event payload")
    _reject_target_outcomes(payload["payload"])
    event_sha = payload.get("event_sha256")
    _require(_is_sha256(event_sha), "event_sha256 must be lowercase SHA-256")
    _require(
        event_sha == _canonical_sha256(payload, omitted="event_sha256"),
        "journal event checksum mismatch",
    )
    return payload


def _empty_journal_state() -> dict[str, Any]:
    return {
        "active_execution_id": None,
        "recovery_active": False,
        "seen_execution_ids": [],
        "completed_execution_ids": [],
        "aborted_execution_ids": [],
    }


def _advance_journal_state(
    state: dict[str, Any],
    event: Mapping[str, Any],
    *,
    event_index: int,
) -> None:
    event_type = str(event["event_type"])
    execution_id = event.get("execution_id")
    if event_index == 0:
        _require(
            event_type == "session_started",
            "journal must begin with session_started",
        )
        _require(execution_id is None, "session_started must not name an execution")
        return

    _require(event_type != "session_started", "session_started may appear only once")
    active = state["active_execution_id"]
    if event_type == "execution_started":
        _require(execution_id is not None, "execution_started requires execution_id")
        _require(active is None, "another execution is already active")
        _require(
            execution_id not in state["seen_execution_ids"],
            "an execution may not be restarted under the same registered ID",
        )
        state["seen_execution_ids"].append(execution_id)
        state["active_execution_id"] = execution_id
        return

    if event_type in {"execution_completed", "execution_aborted"}:
        _require(execution_id is not None, f"{event_type} requires execution_id")
        _require(active is not None, f"{event_type} requires an active execution")
        _require(execution_id == active, f"{event_type} names the wrong execution")
        destination = (
            "completed_execution_ids"
            if event_type == "execution_completed"
            else "aborted_execution_ids"
        )
        state[destination].append(execution_id)
        state["active_execution_id"] = None
        return

    if event_type == "recovery_started":
        _require(not state["recovery_active"], "recovery is already active")
        state["recovery_active"] = True
        return

    if event_type == "recovery_completed":
        _require(state["recovery_active"], "recovery_completed lacks recovery_started")
        state["recovery_active"] = False
        return

    if event_type in _FINAL_EVENT_TYPES:
        _require(execution_id is None, f"{event_type} must not name an execution")
        _require(active is None, "session cannot end while an execution is active")
        _require(
            not state["recovery_active"],
            "session cannot end while recovery is active",
        )
        if event_type == "session_completed":
            _require(
                bool(state["completed_execution_ids"]),
                "session_completed requires at least one completed execution",
            )
        return

    if execution_id is not None:
        _require(
            execution_id in state["seen_execution_ids"],
            "journal event names an execution that was not started",
        )


def _journal_state(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state = _empty_journal_state()
    for event_index, event in enumerate(events):
        _advance_journal_state(state, event, event_index=event_index)
    return state


def build_journal_event(
    *,
    protocol_id: str,
    session_id: str,
    event_type: str,
    source: str,
    sequence: int,
    previous_event_sha256: str | None,
    execution_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
    recorded_at_utc: str | None = None,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Build one content-addressed event for an append-only acquisition journal."""

    event: dict[str, Any] = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "artifact_kind": JOURNAL_EVENT_KIND,
        "protocol_id": protocol_id,
        "session_id": session_id,
        "execution_id": execution_id,
        "sequence": sequence,
        "recorded_at_utc": recorded_at_utc or _utc_now(),
        "monotonic_ns": time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
        "event_type": event_type,
        "source": source,
        "payload": dict(payload or {}),
        "previous_event_sha256": previous_event_sha256,
    }
    event["event_sha256"] = _canonical_sha256(event, omitted="event_sha256")
    return _validate_event_shape(event)
