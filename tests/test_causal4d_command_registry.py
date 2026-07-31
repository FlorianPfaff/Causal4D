from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from causal4d.cli import root
from causal4d.cli.command_registry import (
    CommandSpec,
    find_command,
    grouped_commands,
)


def test_grouped_registry_has_unique_routes_and_legacy_names() -> None:
    commands = grouped_commands()
    assert len({command.route for command in commands}) == len(commands)
    legacy = [
        command.legacy_name
        for command in commands
        if command.legacy_name is not None
    ]
    assert len(set(legacy)) == len(legacy)
    assert find_command("benchmark/counterfactual").target.endswith(
        "counterfactual_benchmark:main"
    )


def test_root_help_and_inventory_do_not_import_command_modules(capsys) -> None:
    assert root.main(["--help"]) == 0
    help_output = capsys.readouterr().out
    assert "usage: causal4d" in help_output
    assert "benchmark" in help_output

    assert root.main(["commands", "list", "--json"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert any(item["route"] == ["protocol", "real"] for item in inventory)


def test_grouped_route_forwards_remaining_arguments(monkeypatch) -> None:
    received: list[str] = []

    def command_main(arguments):
        received.extend(arguments)
        return 7

    monkeypatch.setattr(
        root,
        "import_module",
        lambda name: SimpleNamespace(main=command_main),
    )
    result = root.main(["benchmark", "counterfactual", "--output-dir", "result"])
    assert result == 7
    assert received == ["--output-dir", "result"]


def test_no_argument_main_receives_passthrough_via_sys_argv(monkeypatch) -> None:
    received: list[str] = []
    original_argv = sys.argv

    def command_main():
        received.extend(sys.argv)
        return 5

    monkeypatch.setattr(
        root,
        "import_module",
        lambda name: SimpleNamespace(main=command_main),
    )
    result = root.main(["benchmark", "dynamic-contact", "--output-dir", "result"])
    assert result == 5
    assert received == [
        "causal4d-dynamic-contact-benchmark",
        "--output-dir",
        "result",
    ]
    assert sys.argv is original_argv


def test_describe_and_migrate_report_compatibility_route(capsys) -> None:
    assert root.main(["commands", "describe", "protocol/real", "--json"]) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["legacy_name"] == "causal4d-real-protocol"

    assert root.main(["commands", "migrate", "causal4d-real-protocol", "--json"]) == 0
    migration = json.loads(capsys.readouterr().out)
    assert migration["route"] == ["protocol", "real"]


def test_command_spec_rejects_invalid_routes() -> None:
    with pytest.raises(ValueError, match="non-option"):
        CommandSpec(
            route=("--bad",),
            target="module:main",
            summary="invalid",
            lifecycle="stable",
        )
