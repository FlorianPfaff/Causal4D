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
    validate_runtime_command_inventory,
)


def test_grouped_registry_has_unique_routes_and_legacy_names() -> None:
    commands = grouped_commands()
    assert len({command.route for command in commands}) == len(commands)
    legacy = [
        command.legacy_name for command in commands if command.legacy_name is not None
    ]
    assert len(set(legacy)) == len(legacy)
    assert find_command("benchmark/counterfactual").target.endswith(
        "counterfactual_benchmark:main"
    )
    assert find_command("protocol/acquisition").target.endswith(
        "acquisition_operations:main"
    )


def test_root_help_and_inventory_do_not_import_command_modules(capsys) -> None:
    assert root.main(["--help"]) == 0
    help_output = capsys.readouterr().out
    assert "usage: causal4d" in help_output
    assert "acquisition" in help_output

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
    result = root.main(["protocol", "acquisition", "doctor", "protocol.json"])
    assert result == 7
    assert received == ["doctor", "protocol.json"]


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


def test_runtime_inventory_detects_missing_or_mismatched_grouped_legacy(
    monkeypatch,
) -> None:
    entry_points = []
    for command in grouped_commands():
        if command.legacy_name is None:
            continue
        target = command.target
        if command.legacy_name == "causal4d-real-protocol":
            target = "wrong.module:main"
        if command.legacy_name == "causal4d-real-calibration":
            continue
        entry_points.append(
            SimpleNamespace(
                group="console_scripts",
                name=command.legacy_name,
                value=target,
            )
        )
    entry_points.append(
        SimpleNamespace(
            group="console_scripts",
            name="causal4d-unmapped-research-command",
            value="module:main",
        )
    )
    monkeypatch.setattr(
        "causal4d.cli.command_registry.importlib.metadata.distribution",
        lambda name: SimpleNamespace(entry_points=entry_points),
    )
    report = validate_runtime_command_inventory(require_installed=True)
    assert report["valid"] is False
    assert report["missing_grouped_legacy_executables"] == ["causal4d-real-calibration"]
    assert report["target_mismatches"][0]["legacy_name"] == "causal4d-real-protocol"
    assert report["ungrouped_legacy_executables"] == [
        "causal4d-unmapped-research-command"
    ]


def test_commands_validate_reports_source_checkout_without_install(
    monkeypatch, capsys
) -> None:
    def missing(name):
        raise root.metadata.PackageNotFoundError

    monkeypatch.setattr(
        "causal4d.cli.command_registry.importlib.metadata.distribution",
        missing,
    )
    assert root.main(["commands", "validate", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["installed_distribution_present"] is False
    assert report["valid"] is True
    assert root.main(["commands", "validate", "--json", "--require-installed"]) == 2


def test_command_spec_rejects_invalid_routes() -> None:
    with pytest.raises(ValueError, match="non-option"):
        CommandSpec(
            route=("--bad",),
            target="module:main",
            summary="invalid",
            lifecycle="stable",
        )
