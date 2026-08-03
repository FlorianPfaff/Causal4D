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
        "       causal4d commands {list,describe,migrate,validate} ...",
        "       causal4d legacy <historical-suffix> [arguments]",
        "",
        "Grouped command surface for Causal4D. Command modules are imported lazily.",
        "",
        "groups:",
    ]
    for group, commands in groups.items():
        lines.append(f"  {group}")
        for command in commands:
            suffix = " ".join(command.route[1:])
            lines.append(f"    {suffix:<24} {command.summary}")
    lines.extend(
        (
            "",
            "introspection:",
            "  commands list [--json] [--include-legacy]",
            "  commands describe <route-or-legacy-name> [--json]",
            "  commands migrate <historical-executable> [--json]",
            "  commands validate [--json] [--require-installed]",
            "",
            "Use 'causal4d <group> <command> --help' for command-specific help.",
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
        sys.argv = [command.legacy_name or "causal4d", *arguments]
        try:
            result = function()
        finally:
            sys.argv = original_argv
    return int(result or 0)


def _commands_list(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="causal4d commands list")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--include-legacy", action="store_true")
    parsed = parser.parse_args(arguments)
    inventory = command_inventory(include_legacy=parsed.include_legacy)
    if parsed.json:
        print(json.dumps([command.as_dict() for command in inventory], indent=2))
    else:
        for command in inventory:
            legacy = f" [{command.legacy_name}]" if command.legacy_name else ""
            print(f"{command.route_name:<38} {command.lifecycle:<12}{legacy}")
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
        print(f"target: {command.target}")
        print(f"lifecycle: {command.lifecycle}")
        print(f"summary: {command.summary}")
        if command.legacy_name:
            print(f"legacy executable: {command.legacy_name}")
        if command.extras:
            print(f"optional extras: {', '.join(command.extras)}")
    return 0


def _commands_migrate(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="causal4d commands migrate")
    parser.add_argument("legacy_name")
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(arguments)
    requested = parsed.legacy_name
    if not requested.startswith("causal4d-"):
        requested = f"causal4d-{requested}"
    matches = [
        command for command in grouped_commands() if command.legacy_name == requested
    ]
    if not matches:
        parser.error(f"no grouped route is registered for {requested}")
    command = matches[0]
    payload = {"legacy_name": requested, "route": list(command.route)}
    if parsed.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{requested} -> causal4d {command.route_name}")
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
        print(f"installed distribution: {report['installed_distribution_present']}")
        print(f"ungrouped legacy executables: {report['ungrouped_legacy_count']}")
        missing = report["missing_grouped_legacy_executables"]
        if missing:
            print("missing grouped legacy executables: " + ", ".join(missing))
        mismatches = report["target_mismatches"]
        for mismatch in mismatches:
            print(
                "target mismatch: "
                f"{mismatch['legacy_name']} -> {mismatch['installed_target']} "
                f"(registered {mismatch['registered_target']})"
            )
    return 0 if report["valid"] else 2


def _commands(arguments: Sequence[str]) -> int:
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "usage: causal4d commands {list,describe,migrate,validate} ...\n\n"
            "Inspect the typed command registry and legacy migrations."
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


def _resolve_route(arguments: Sequence[str]) -> tuple[CommandSpec, list[str]]:
    for command in sorted(
        command_inventory(include_legacy=True),
        key=lambda item: len(item.route),
        reverse=True,
    ):
        width = len(command.route)
        if tuple(arguments[:width]) == command.route:
            return command, list(arguments[width:])
    raise KeyError(" ".join(arguments))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_root_help())
        return 0
    if arguments[0] == "--version":
        print(_version())
        return 0
    if arguments[0] == "commands":
        return _commands(arguments[1:])
    try:
        command, remaining = _resolve_route(arguments)
    except KeyError:
        print(_root_help(), file=sys.stderr)
        print(f"\nerror: unknown command route: {' '.join(arguments)}", file=sys.stderr)
        return 2
    return _invoke(command, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
