"""Durable append and validation for the acquisition journal."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from causal4d.acquisition_flight_common import (
    JOURNAL_SCHEMA_VERSION,
    _assert_no_symlink_components,
    _assert_ordinary_file_or_missing,
    _canonical_json_bytes,
    _fsync_directory,
    _require,
    _sha256_file,
    journal_seal_path,
)
from causal4d.acquisition_journal_model import (
    _FINAL_EVENT_TYPES,
    _advance_journal_state,
    _empty_journal_state,
    _journal_state,
    _validate_event_shape,
    build_journal_event,
)


def _last_nonempty_line(handle: Any) -> str | None:
    handle.seek(0, os.SEEK_END)
    position = handle.tell()
    if position == 0:
        return None
    block_size = 8192
    data = b""
    while position > 0:
        read_size = min(block_size, position)
        position -= read_size
        handle.seek(position)
        data = handle.read(read_size) + data
        lines = data.splitlines()
        complete_lines = lines if position == 0 else lines[1:]
        for line in reversed(complete_lines):
            if not line.strip():
                continue
            try:
                return line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("journal ends with invalid UTF-8") from error
    return None


def append_journal_event(
    journal_path: str | Path,
    *,
    protocol_id: str,
    session_id: str,
    event_type: str,
    source: str,
    execution_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
    recorded_at_utc: str | None = None,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Append one fsync'ed, hash-chained event without replacing prior bytes."""

    path = Path(journal_path)
    seal = journal_seal_path(path)
    _assert_ordinary_file_or_missing(path, name="acquisition journal")
    _assert_ordinary_file_or_missing(seal, name="acquisition journal seal")
    _require(not seal.exists(), "acquisition journal is sealed")
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(path.parent, name="acquisition journal parent")

    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "r+b") as handle:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows fallback
                fcntl = None
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _require(not seal.exists(), "acquisition journal is sealed")
            last_line = _last_nonempty_line(handle)
            if last_line is None:
                sequence = 0
                previous = None
                state = _empty_journal_state()
            else:
                validation = validate_acquisition_journal(path)
                try:
                    last = _validate_event_shape(json.loads(last_line))
                except json.JSONDecodeError as error:
                    raise ValueError("journal ends with invalid JSON") from error
                _require(
                    last["event_sha256"] == validation["final_event_sha256"],
                    "journal tail differs from the validated hash chain",
                )
                _require(last["protocol_id"] == protocol_id, "journal protocol changed")
                _require(last["session_id"] == session_id, "journal session changed")
                sequence = int(last["sequence"]) + 1
                previous = str(last["event_sha256"])
                state = {
                    "active_execution_id": validation["active_execution_id"],
                    "recovery_active": validation["recovery_active"],
                    "seen_execution_ids": list(validation["seen_execution_ids"]),
                    "completed_execution_ids": list(
                        validation["completed_execution_ids"]
                    ),
                    "aborted_execution_ids": list(
                        validation["aborted_execution_ids"]
                    ),
                }
                _require(
                    last["event_type"] not in _FINAL_EVENT_TYPES,
                    "cannot append after a terminal session event",
                )
                if monotonic_ns is not None:
                    _require(
                        monotonic_ns >= int(last["monotonic_ns"]),
                        "journal monotonic clock moved backward",
                    )
            event = build_journal_event(
                protocol_id=protocol_id,
                session_id=session_id,
                execution_id=execution_id,
                event_type=event_type,
                source=source,
                sequence=sequence,
                previous_event_sha256=previous,
                payload=payload,
                recorded_at_utc=recorded_at_utc,
                monotonic_ns=monotonic_ns,
            )
            _advance_journal_state(state, event, event_index=sequence)
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    _fsync_directory(path.parent)
    return event


def validate_acquisition_journal(journal_path: str | Path) -> dict[str, Any]:
    """Validate every event, the hash chain, and the session-level invariants."""

    path = Path(journal_path)
    _assert_ordinary_file_or_missing(path, name="acquisition journal")
    _require(path.is_file(), "acquisition journal does not exist")
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"blank journal line at {line_number}")
            try:
                event = _validate_event_shape(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid journal JSON at line {line_number}"
                ) from error
            if not events:
                _require(event["sequence"] == 0, "journal must begin at sequence zero")
                _require(
                    event["previous_event_sha256"] is None,
                    "first journal event must not have a predecessor",
                )
                _require(
                    event["event_type"] == "session_started",
                    "journal must begin with session_started",
                )
            else:
                previous = events[-1]
                _require(
                    event["sequence"] == previous["sequence"] + 1,
                    f"journal sequence gap at line {line_number}",
                )
                _require(
                    event["previous_event_sha256"] == previous["event_sha256"],
                    f"journal hash-chain mismatch at line {line_number}",
                )
                _require(
                    event["protocol_id"] == previous["protocol_id"],
                    "journal protocol changed",
                )
                _require(
                    event["session_id"] == previous["session_id"],
                    "journal session changed",
                )
                _require(
                    event["monotonic_ns"] >= previous["monotonic_ns"],
                    f"monotonic clock moved backward at line {line_number}",
                )
                _require(
                    previous["event_type"] not in _FINAL_EVENT_TYPES,
                    "journal contains events after terminal session event",
                )
            events.append(event)
    _require(events, "acquisition journal is empty")
    state = _journal_state(events)
    digest, size = _sha256_file(path)
    first = events[0]
    last = events[-1]
    execution_ids = sorted(
        {
            str(event["execution_id"])
            for event in events
            if event.get("execution_id") is not None
        }
    )
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "artifact_kind": "Causal4DAcquisitionJournalValidation",
        "valid": True,
        "protocol_id": first["protocol_id"],
        "session_id": first["session_id"],
        "event_count": len(events),
        "execution_ids": execution_ids,
        "seen_execution_ids": list(state["seen_execution_ids"]),
        "completed_execution_ids": list(state["completed_execution_ids"]),
        "aborted_execution_ids": list(state["aborted_execution_ids"]),
        "active_execution_id": state["active_execution_id"],
        "recovery_active": state["recovery_active"],
        "first_event_type": first["event_type"],
        "last_event_type": last["event_type"],
        "first_recorded_at_utc": first["recorded_at_utc"],
        "last_recorded_at_utc": last["recorded_at_utc"],
        "final_event_sha256": last["event_sha256"],
        "journal_sha256": digest,
        "journal_bytes": size,
        "terminal": last["event_type"] in _FINAL_EVENT_TYPES,
        "target_outcomes_used": False,
    }
