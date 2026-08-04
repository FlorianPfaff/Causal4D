from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import tempfile

import pytest

from causal4d import acquisition_journal_lock
from causal4d.acquisition_flight_recorder import (
    append_journal_event,
    journal_seal_path,
    seal_acquisition_journal,
    validate_acquisition_journal,
)


UTC = "2026-08-05T08:00:00+00:00"


def _append(
    path: Path,
    event_type: str,
    *,
    monotonic_ns: int | None = None,
) -> dict[str, object]:
    return append_journal_event(
        path,
        protocol_id="protocol-v1",
        session_id="session-1",
        event_type=event_type,
        source="locking-test",
        recorded_at_utc=UTC,
        monotonic_ns=monotonic_ns,
    )


def _append_operator_note(path: str, worker_index: int) -> int:
    event = append_journal_event(
        path,
        protocol_id="protocol-v1",
        session_id="session-1",
        event_type="operator_note",
        source=f"worker-{worker_index}",
        payload={"worker_index": worker_index},
        recorded_at_utc=UTC,
    )
    return int(event["sequence"])


def test_append_fails_before_creating_a_journal_without_locking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "acquisition.jsonl"
    monkeypatch.setattr(acquisition_journal_lock, "_fcntl", None)

    with pytest.raises(RuntimeError, match="require POSIX advisory file locking"):
        _append(journal, "session_started", monotonic_ns=10)

    assert not journal.exists()


def test_seal_fails_without_publishing_when_locking_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "acquisition.jsonl"
    _append(journal, "session_started", monotonic_ns=10)
    _append(journal, "session_completed", monotonic_ns=20)
    monkeypatch.setattr(acquisition_journal_lock, "_fcntl", None)

    with pytest.raises(RuntimeError, match="require POSIX advisory file locking"):
        seal_acquisition_journal(journal, sealed_by="operator.primary")

    assert not journal_seal_path(journal).exists()
    assert validate_acquisition_journal(journal)["event_count"] == 2


def test_lock_is_released_when_the_transaction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingLockBackend:
        LOCK_EX = 1
        LOCK_UN = 2

        def __init__(self) -> None:
            self.operations: list[int] = []

        def flock(self, file_descriptor: int, operation: int, /) -> None:
            assert file_descriptor >= 0
            self.operations.append(operation)

    backend = RecordingLockBackend()
    monkeypatch.setattr(acquisition_journal_lock, "_fcntl", backend)

    with tempfile.TemporaryFile(mode="w+b") as handle:
        with pytest.raises(RuntimeError, match="transaction failed"):
            with acquisition_journal_lock.exclusive_acquisition_journal_lock(handle):
                raise RuntimeError("transaction failed")

    assert backend.operations == [backend.LOCK_EX, backend.LOCK_UN]


@pytest.mark.skipif(
    not acquisition_journal_lock.acquisition_journal_locking_available(),
    reason="POSIX advisory locking is unavailable",
)
def test_concurrent_process_appends_form_one_contiguous_hash_chain(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "acquisition.jsonl"
    _append(journal, "session_started", monotonic_ns=10)
    worker_count = 8
    context = get_context("spawn")

    with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        sequences = list(
            executor.map(
                _append_operator_note,
                [str(journal)] * worker_count,
                range(worker_count),
            )
        )

    assert sorted(sequences) == list(range(1, worker_count + 1))
    validation = validate_acquisition_journal(journal)
    assert validation["event_count"] == worker_count + 1
    assert validation["final_event_sha256"]
    assert validation["terminal"] is False
