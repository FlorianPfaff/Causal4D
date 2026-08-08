from __future__ import annotations

import json

import pytest

import causal4d.atomic_io as atomic_io_module
from causal4d.atomic_io import (
    atomic_write_binary,
    atomic_write_json,
    atomic_write_text,
)


def test_atomic_write_json_publishes_sorted_finite_json(tmp_path) -> None:
    target = tmp_path / "nested" / "artifact.json"
    atomic_write_json(target, {"b": 2, "a": 1})

    assert target.read_text(encoding="utf-8") == ('{\n  "a": 1,\n  "b": 2\n}\n')
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_atomic_write_json_preserves_destination_on_serialization_failure(
    tmp_path,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("original\n", encoding="utf-8")

    with pytest.raises(ValueError):
        atomic_write_json(target, {"bad": float("nan")})

    assert target.read_text(encoding="utf-8") == "original\n"


def test_atomic_write_json_can_fail_closed_on_existing_destination(tmp_path) -> None:
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"version": 1}, overwrite=False)

    with pytest.raises(FileExistsError):
        atomic_write_json(target, {"version": 2}, overwrite=False)

    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_atomic_write_binary_validates_before_publication(tmp_path) -> None:
    target = tmp_path / "artifact.bin"

    def writer(handle) -> None:
        handle.write(b"candidate")

    def reject(path) -> None:
        assert path.read_bytes() == b"candidate"
        raise ValueError("invalid candidate")

    with pytest.raises(ValueError, match="invalid candidate"):
        atomic_write_binary(target, writer, validate=reject)

    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_atomic_write_binary_can_fail_closed_on_existing_destination(tmp_path) -> None:
    target = tmp_path / "artifact.bin"
    atomic_write_binary(target, lambda handle: handle.write(b"first"), overwrite=False)

    with pytest.raises(FileExistsError):
        atomic_write_binary(
            target,
            lambda handle: handle.write(b"second"),
            overwrite=False,
        )

    assert target.read_bytes() == b"first"


def test_no_overwrite_publication_rolls_back_failed_directory_durability(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.txt"
    calls = 0

    def fail_first_directory_sync(path) -> None:
        nonlocal calls
        assert path == tmp_path
        calls += 1
        if calls == 1:
            raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(
        atomic_io_module,
        "_fsync_directory",
        fail_first_directory_sync,
    )
    with pytest.raises(OSError, match="simulated directory fsync failure"):
        atomic_write_text(target, "candidate", overwrite=False)

    assert calls == 2
    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_overwrite_publication_retains_visible_target_on_durability_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.txt"
    target.write_text("old", encoding="utf-8")

    def fail_directory_sync(path) -> None:
        assert path == tmp_path
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(
        atomic_io_module,
        "_fsync_directory",
        fail_directory_sync,
    )
    with pytest.raises(OSError, match="simulated directory fsync failure"):
        atomic_write_text(target, "candidate", overwrite=True)

    assert target.read_text(encoding="utf-8") == "candidate"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
