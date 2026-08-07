from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from causal4d import stack_lock
from causal4d.cli import root
from causal4d.cli.stack import main as stack_main


REVISIONS = {
    "prob4d": "1" * 40,
    "bayesian-phystwin": "2" * 40,
    "causal4d": "3" * 40,
}


def _write_wheel(tmp_path: Path, name: str, version: str) -> Path:
    token = name.replace("-", "_")
    path = tmp_path / f"{token}-{version}-py3-none-any.whl"
    dist_info = f"{token}-{version}.dist-info"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            (
                "Metadata-Version: 2.1\n"
                f"Name: {name}\n"
                f"Version: {version}\n"
            ),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )
    return path


def _stack_wheels(tmp_path: Path) -> dict[str, Path]:
    return {
        "prob4d": _write_wheel(tmp_path, "prob4d", "0.3.0"),
        "bayesian-phystwin": _write_wheel(
            tmp_path,
            "bayesian-phystwin",
            "0.4.0",
        ),
        "causal4d": _write_wheel(tmp_path, "causal4d", "0.5.0"),
    }


def test_stack_lock_is_deterministic_and_pipeline_ordered(tmp_path: Path) -> None:
    wheels = _stack_wheels(tmp_path)
    paths = list(reversed(tuple(wheels.values())))

    first = stack_lock.build_stack_lock(
        paths,
        source_revisions=REVISIONS,
    )
    second = stack_lock.build_stack_lock(
        paths,
        source_revisions=REVISIONS,
    )

    assert first == second
    assert [item["name"] for item in first["distributions"]] == list(
        stack_lock.STACK_PIPELINE
    )
    assert len(first["lock_id"]) == 64
    assert stack_lock.validate_stack_lock(first) == first


def test_stack_lock_roundtrip_rejects_tampering_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    wheels = _stack_wheels(tmp_path)
    lock = stack_lock.build_stack_lock(
        list(wheels.values()),
        source_revisions=REVISIONS,
    )
    path = tmp_path / "stack-lock.json"
    stack_lock.write_stack_lock(path, lock)

    assert stack_lock.load_stack_lock(path) == lock

    tampered = deepcopy(lock)
    tampered["distributions"][0]["wheel"]["size_bytes"] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="lock_id does not match"):
        stack_lock.load_stack_lock(path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_name":"first","schema_name":"second"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        stack_lock.load_stack_lock(duplicate)


def test_stack_lock_verifies_exact_wheels_and_detects_drift(tmp_path: Path) -> None:
    wheels = _stack_wheels(tmp_path)
    lock = stack_lock.build_stack_lock(
        list(wheels.values()),
        source_revisions=REVISIONS,
    )

    report = stack_lock.verify_stack_lock(
        lock,
        wheel_paths=list(wheels.values()),
    )
    assert report["valid"] is True
    assert report["wheel_set"]["verified"] is True

    renamed = tmp_path / "downloaded-prob4d.whl"
    wheels["prob4d"].rename(renamed)
    wheels["prob4d"] = renamed
    assert stack_lock.verify_stack_lock(
        lock,
        wheel_paths=list(wheels.values()),
    )["valid"]

    with wheels["causal4d"].open("ab") as handle:
        handle.write(b"tamper")
    drifted = stack_lock.verify_stack_lock(
        lock,
        wheel_paths=list(wheels.values()),
    )
    assert drifted["valid"] is False
    assert any(
        "wheel identity mismatch for causal4d" in item
        for item in drifted["errors"]
    )


def test_stack_cli_creates_and_verifies_lock(tmp_path: Path, capsys) -> None:
    wheels = _stack_wheels(tmp_path)
    output = tmp_path / "stack-lock.json"
    create_arguments = ["create"]
    for path in wheels.values():
        create_arguments.extend(("--wheel", str(path)))
    for name, revision in REVISIONS.items():
        create_arguments.extend(("--revision", f"{name}={revision}"))
    create_arguments.extend(("--output", str(output), "--json"))

    assert stack_main(create_arguments) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["lock_id"] == stack_lock.load_stack_lock(output)["lock_id"]

    verify_arguments = ["verify", "--lock", str(output), "--json"]
    for path in wheels.values():
        verify_arguments.extend(("--wheel", str(path)))
    assert stack_main(verify_arguments) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True

    assert stack_main(
        ["verify", "--lock", str(output), "--lock-only", "--json"]
    ) == 0
    lock_only = json.loads(capsys.readouterr().out)
    assert lock_only["valid"] is True
    assert lock_only["wheel_set"]["verified"] is False


def test_root_routes_stack_operations_lazily(monkeypatch, capsys) -> None:
    assert root.main(["--help"]) == 0
    assert "causal4d stack {create,verify}" in capsys.readouterr().out

    received: list[str] = []

    def command_main(arguments):
        received.extend(arguments)
        return 9

    def fake_import(name: str):
        assert name == "causal4d.cli.stack"
        return SimpleNamespace(main=command_main)

    monkeypatch.setattr(root, "import_module", fake_import)
    result = root.main(["stack", "verify", "--lock", "lock.json", "--lock-only"])
    assert result == 9
    assert received == ["verify", "--lock", "lock.json", "--lock-only"]
