from __future__ import annotations

import json

import pytest

from causal4d.atomic_io import atomic_write_json


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
