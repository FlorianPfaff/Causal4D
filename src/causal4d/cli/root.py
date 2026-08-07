"""Grouped, lazy command-line entry point for Causal4D."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module, metadata
import inspect
import json
import sys

from causal4d.cli.command_registry import (
    CommandSpec,
    command_inventory,
    find_command,
    grouped_commands,
    historical_commands,
    validate_runtime_command_inventory,
)


def _version() -> str:
    try:
        return metadata.version("causal4d")
    except metadata.PackageNotFoundError:
        return "unknown"


def _root_help() -> str:
    groups: dict[str, list[CommandSpec]] = {}
    for command in grouped_commands():
        groups.setdefault(command.route[0], []).append(command)
    lines = [
        "usage: causal4d <group> <command> [arguments]",
        "       causal4d stack {create,verify} ...",
        "       causal4d commands {list,describe,migrate,validate} ...",
        "",
        "Single executable for all Causal4D commands. Modules are imported lazily.",
        "",
        "groups:",
    ]
    for group, commands in groups.items():
        lines.append(f"  {group}")
        for command in commands:
            suffix = " ".join(command.route[1:])
            lines.append(f"    {suffix:<40} {command.summary}")
    lines.extend(
        (
            "",
            "stack locks:",
            "  stack create --wheel PATH ... --revision DISTRIBUTION=SHA ...",
            "  stack verify --lock PATH --wheel PATH ... [--json]",
            "  stack verify --lock PATH --lock-only [--json]",
            "",
            "introspection:",
            "  commands list [--json] [--removed-only]",
            "  commands describe <route-or-removed-executable> [--json]",
            "  commands migrate <removed-executable> [--json]",
            "  commands validate [--json] [--require-installed]",
            "",
            "Historical causal4d-* executables were removed in 0.5.0.",
            "Use 'causal4d commands migrate <old-name>' for the successor route.",
        )
    )
    return "\n".join(lines)


def _invoke(command: CommandSpec, arguments: Sequence[str]) -> int:
    module_name, function_name = command.target.split(":", 1)
    function = getattr(import_module(module_name), function_name)
    parameters = tuple(inspect.signature(function).parameters.values())
    accepts_argv = any(
        parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        for parameter in parameters
    )
    if accepts_argv:
        result = function(list(arguments))
    else:
        original_argv = sys.argv
        sys.argv = [command.invocation_text, *arguments]
        try:
            result = function()
        finally:
            sys.argv = original_argv
    return int(result or 0)


def _commands_list(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="causal4d commands list")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--removed-only", action="store_true")
    parsed = parser.parse_args(arguments)
    inventory = command_inventory(removed_only=parsed.removed_only)
    if parsed.json:
        print(json.dumps([command.as_dict() for command in inventory], indent=2))
    else:
        for command in inventory:
            historical = (
                f" [removed: {command.historical_name}]"
                if command.historical_name
                else ""
            )
            print(f"{command.route_name:<52} {command.lifecycle:<12}{historical}")
    return 0


def _commands_describe(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="causal4d commands describe")
    parser.add_argument("name", nargs="+")
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(arguments)
    name = " ".join(parsed.name)
    try:
        command = find_command(name)
    except KeyError:
        parser.error(f"unknown command: {name}")
    if parsed.json:
        print(json.dumps(command.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"route: {command.route_name}")
        print(f"invocation: {command.invocation_text}")
        print(f"target: {command.target}")
        print(f"lifecycle: {command.lifecycle}")
        print(f"claim-bearing: {str(command.claim_bearing).lower()}")
        print(f"summary: {command.summary}")
        if command.historical_name:
            print(
                "removed executable: "
                f"{command.historical_name} (removed in {command.removed_in})"
            )
        if command.extras:
            print(f"optional extras: {', '.join(command.extras)}")
        if command.requires:
            print(f"external requirements: {', '.join(command.requires)}")
    return 0


def _commands_migrate(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="causal4d commands migrate")
    parser.add_argument("historical_name")
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(arguments)
    requested = parsed.historical_name
    if not requested.startswith("causal4d-"):
        requested = f"causal4d-{requested}"
    matches = [
        command
        for command in historical_commands()
        if command.historical_name == requested
    ]
    if not matches:
        parser.error(f"no successor route is registered for {requested}")
    command = matches[0]
    payload = {
        "historical_name": requested,
        "removed_in": command.removed_in,
        "route": list(command.route),
        "invocation": list(command.invocation),
        "invocation_text": command.invocation_text,
    }
    if parsed.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{requested} -> {command.invocation_text}")
    return 0


def _commands_validate(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="causal4d commands validate")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-installed", action="store_true")
    parsed = parser.parse_args(arguments)
    report = validate_runtime_command_inventory(
        require_installed=parsed.require_installed
    )
    if parsed.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        state = "valid" if report["valid"] else "invalid"
        print(f"command inventory: {state}")
        print(f"grouped routes: {report['grouped_route_count']}")
        print(f"historical mappings: {report['historical_executable_count']}")
        print(f"installed distribution: {report['installed_distribution_present']}")
        for name in report["missing_console_scripts"]:
            print(f"missing console script: {name}")
        for name in report["unexpected_console_scripts"]:
            print(f"unexpected console script: {name}")
        for name in report["target_mismatches"]:
            print(f"console-script target mismatch: {name}")
        for name in report["removed_historical_executables_installed"]:
            print(f"removed historical executable still installed: {name}")
    return 0 if report["valid"] else 2


def _commands(arguments: Sequence[str]) -> int:
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "usage: causal4d commands {list,describe,migrate,validate} ...\n\n"
            "Inspect current routes and removed executable migrations."
        )
        return 0
    operation, *remaining = arguments
    if operation == "list":
        return _commands_list(remaining)
    if operation == "describe":
        return _commands_describe(remaining)
    if operation == "migrate":
        return _commands_migrate(remaining)
    if operation == "validate":
        return _commands_validate(remaining)
    raise SystemExit(f"unknown commands operation: {operation}")


def _stack(arguments: Sequence[str]) -> int:
    function = getattr(import_module("causal4d.cli.stack"), "main")
    return int(function(list(arguments)) or 0)


def _resolve_route(arguments: Sequence[str]) -> tuple[CommandSpec, list[str]]:
    for command in sorted(
        grouped_commands(),
        key=lambda item: len(item.route),
        reverse=True,
    ):
        width = len(command.route)
        if tuple(arguments[:width]) == command.route:
            return command, list(arguments[width:])
    raise KeyError(" ".join(arguments))


def _print_removed_migration(name: str) -> bool:
    try:
        command = find_command(name)
    except KeyError:
        return False
    if command.historical_name != name:
        return False
    print(
        f"error: {name} was removed in {command.removed_in}; "
        f"use '{command.invocation_text}'",
        file=sys.stderr,
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_root_help())
        return 0
    if arguments[0] == "--version":
        print(_version())
        return 0
    if arguments[0] == "stack":
        return _stack(arguments[1:])
    if arguments[0] == "commands":
        return _commands(arguments[1:])
    if arguments[0].startswith("causal4d-") and _print_removed_migration(arguments[0]):
        return 2
    try:
        command, remaining = _resolve_route(arguments)
    except KeyError:
        print(_root_help(), file=sys.stderr)
        print(f"\nerror: unknown command route: {' '.join(arguments)}", file=sys.stderr)
        return 2
    return _invoke(command, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
