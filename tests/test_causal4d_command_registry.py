from __future__ import annotations

import importlib.metadata
import json
import sys
from types import SimpleNamespace

import pytest

from causal4d.cli import root
from causal4d.cli.command_registry import (
    CommandSpec,
    find_command,
    grouped_commands,
    historical_commands,
    validate_runtime_command_inventory,
)


def test_grouped_registry_is_complete_and_unique() -> None:
    commands = grouped_commands()
    assert len({command.route for command in commands}) == len(commands)
    historical = historical_commands()
    assert len(historical) == 67
    assert len({command.historical_name for command in historical}) == 67
    assert find_command("benchmark/counterfactual").target.endswith(
        "counterfactual_benchmark:main"
    )
    assert find_command("causal4d-counterfactual-benchmark").route == (
        "benchmark",
        "counterfactual",
    )
    assert all(
        command.historical_name is None or command.removed_in == "0.5.0"
        for command in commands
    )


def test_root_help_and_inventory_do_not_import_command_modules(capsys) -> None:
    assert root.main(["--help"]) == 0
    help_output = capsys.readouterr().out
    assert "Single executable" in help_output
    assert "Historical causal4d-* executables were removed" in help_output
    assert "legacy <historical-suffix>" not in help_output

    assert root.main(["commands", "list", "--json"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert any(item["route"] == ["protocol", "real"] for item in inventory)
    assert all(item["invocation"][0] == "causal4d" for item in inventory)


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


def test_no_argument_main_receives_grouped_prog_via_sys_argv(monkeypatch) -> None:
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
        "causal4d benchmark dynamic-contact",
        "--output-dir",
        "result",
    ]
    assert sys.argv is original_argv


def test_describe_and_migrate_report_removed_executable(capsys) -> None:
    assert root.main(["commands", "describe", "protocol/real", "--json"]) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["historical_name"] == "causal4d-real-protocol"
    assert description["historical_executable_installed"] is False

    assert root.main(["commands", "migrate", "causal4d-real-protocol", "--json"]) == 0
    migration = json.loads(capsys.readouterr().out)
    assert migration["route"] == ["protocol", "real"]
    assert migration["invocation_text"] == "causal4d protocol real"
    assert migration["removed_in"] == "0.5.0"


def test_removed_name_is_not_a_runtime_route(capsys) -> None:
    result = root.main(["causal4d-real-protocol", "--help"])
    captured = capsys.readouterr()
    assert result == 2
    assert "was removed in 0.5.0" in captured.err
    assert "causal4d protocol real" in captured.err


def test_runtime_inventory_requires_only_primary_script(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal4d.cli.command_registry._installed_console_scripts",
        lambda: {"causal4d": "causal4d.cli.root:main"},
    )
    report = validate_runtime_command_inventory(require_installed=True)
    assert report["valid"] is True
    assert report["removed_historical_executables_installed"] == []


def test_runtime_inventory_rejects_historical_wrapper(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal4d.cli.command_registry._installed_console_scripts",
        lambda: {
            "causal4d": "causal4d.cli.root:main",
            "causal4d-real-protocol": "causal4d.cli.real_protocol:main",
        },
    )
    report = validate_runtime_command_inventory(require_installed=True)
    assert report["valid"] is False
    assert report["removed_historical_executables_installed"] == [
        "causal4d-real-protocol"
    ]


def test_distribution_metadata_declares_one_console_script() -> None:
    try:
        distribution = importlib.metadata.distribution("causal4d")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("source-only checkout has no installed distribution metadata")
    scripts = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
        and (entry.name == "causal4d" or entry.name.startswith("causal4d-"))
    }
    assert scripts == {"causal4d": "causal4d.cli.root:main"}


def test_command_spec_rejects_invalid_routes() -> None:
    with pytest.raises(ValueError, match="non-option"):
        CommandSpec(
            route=("--bad",),
            target="module:main",
            summary="invalid",
            lifecycle="stable",
        )
