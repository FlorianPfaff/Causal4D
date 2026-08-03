"""Exactly-once sealing for acquisition journals."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json
from causal4d.acquisition_flight_common import (
    JOURNAL_SCHEMA_VERSION,
    JOURNAL_SEAL_KIND,
    _assert_ordinary_file_or_missing,
    _canonical_sha256,
    _is_sha256,
    _parse_utc,
    _require,
    _utc_now,
    journal_seal_path,
)
from causal4d.acquisition_journal_io import validate_acquisition_journal
from causal4d.acquisition_journal_model import _FINAL_EVENT_TYPES


def seal_acquisition_journal(
    journal_path: str | Path,
    *,
    sealed_by: str,
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Seal a terminal journal exactly once with a portable content identity."""

    _require(
        isinstance(sealed_by, str) and bool(sealed_by.strip()),
        "sealed_by must be nonempty",
    )
    journal = Path(journal_path)
    seal_path = journal_seal_path(journal)
    _assert_ordinary_file_or_missing(journal, name="acquisition journal")
    _require(journal.is_file(), "acquisition journal does not exist")
    _assert_ordinary_file_or_missing(seal_path, name="acquisition journal seal")
    _require(not seal_path.exists(), "acquisition journal seal already exists")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(journal, flags)
    with os.fdopen(descriptor, "rb") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows fallback
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _require(not seal_path.exists(), "acquisition journal seal already exists")
        validation = validate_acquisition_journal(journal)
        last_event_type = str(validation["last_event_type"])
        _require(
            last_event_type in _FINAL_EVENT_TYPES,
            "journal must end with session_completed or session_aborted before sealing",
        )
        timestamp = sealed_at_utc or _utc_now()
        _parse_utc(timestamp, name="sealed_at_utc")
        seal: dict[str, Any] = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "artifact_kind": JOURNAL_SEAL_KIND,
            "status": "sealed",
            "protocol_id": validation["protocol_id"],
            "session_id": validation["session_id"],
            "execution_ids": validation["execution_ids"],
            "event_count": validation["event_count"],
            "journal_sha256": validation["journal_sha256"],
            "journal_bytes": validation["journal_bytes"],
            "final_event_sha256": validation["final_event_sha256"],
            "session_outcome": _FINAL_EVENT_TYPES[last_event_type],
            "sealed_by": sealed_by.strip(),
            "sealed_at_utc": timestamp,
            "target_outcomes_used": False,
        }
        seal["seal_sha256"] = _canonical_sha256(seal, omitted="seal_sha256")
        atomic_write_json(seal_path, seal, overwrite=False)
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return seal


def validate_acquisition_journal_seal(journal_path: str | Path) -> dict[str, Any]:
    """Reopen the deterministic seal and prove that journal bytes did not change."""

    journal = Path(journal_path)
    seal_path = journal_seal_path(journal)
    _assert_ordinary_file_or_missing(seal_path, name="acquisition journal seal")
    _require(seal_path.is_file(), "acquisition journal seal does not exist")
    payload = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "journal seal must be a JSON object")
    seal = dict(payload)
    _require(seal.get("schema_version") == JOURNAL_SCHEMA_VERSION, "wrong seal schema")
    _require(seal.get("artifact_kind") == JOURNAL_SEAL_KIND, "wrong seal kind")
    _require(seal.get("status") == "sealed", "journal seal is not sealed")
    _require(
        seal.get("target_outcomes_used") is False,
        "journal seal used target outcomes",
    )
    _parse_utc(seal.get("sealed_at_utc"), name="sealed_at_utc")
    _require(
        isinstance(seal.get("sealed_by"), str) and bool(seal["sealed_by"].strip()),
        "journal seal signer is missing",
    )
    _require(_is_sha256(seal.get("seal_sha256")), "invalid seal SHA-256")
    _require(
        seal["seal_sha256"] == _canonical_sha256(seal, omitted="seal_sha256"),
        "journal seal checksum mismatch",
    )
    validation = validate_acquisition_journal(journal)
    for field in (
        "protocol_id",
        "session_id",
        "execution_ids",
        "event_count",
        "journal_sha256",
        "journal_bytes",
        "final_event_sha256",
    ):
        _require(
            seal.get(field) == validation.get(field),
            f"journal seal {field} mismatch",
        )
    expected_outcome = _FINAL_EVENT_TYPES.get(str(validation["last_event_type"]))
    _require(
        seal.get("session_outcome") == expected_outcome,
        "journal outcome mismatch",
    )
    return {**seal, "valid": True, "seal_path": str(seal_path)}
