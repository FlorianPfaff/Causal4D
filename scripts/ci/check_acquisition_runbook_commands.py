#!/usr/bin/env python3
"""Reject stale or unknown Causal4D commands in acquisition-critical runbooks.

Causal4D 0.5 installs one ``causal4d`` executable. Historical
``causal4d-*`` executables remain migration metadata only. This check parses
shell command blocks in the operator-facing acquisition runbooks and verifies
that every Causal4D invocation uses either a registered grouped route or one of
the small command-registry management routes.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from causal4d.cli.command_registry import (  # noqa: E402
    PRIMARY_EXECUTABLE,
    grouped_commands,
)

ACQUISITION_RUNBOOKS = (
    Path("README.md"),
    Path("docs/causal4d_preacquisition_readiness.md"),
    Path("docs/causal4d_real_experiment_milestone.md"),
    Path("docs/causal4d_real_evidence_status.md"),
)

SHELL_FENCE_LANGUAGES = frozenset({"bash", "console", "sh", "shell", "zsh"})
COMMAND_MANAGEMENT_ROUTES = frozenset({"describe", "list", "migrate", "validate"})
LEGACY_EXECUTABLE_PATTERN = re.compile(r"^causal4d-[A-Za-z0-9][A-Za-z0-9_-]*$")
SHELL_ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
SHELL_WRAPPERS = frozenset({"command", "env", "sudo", "time"})


@dataclass(frozen=True)
class RunbookCommandIssue:
    """One invalid command found in an operator-facing Markdown runbook."""

    path: Path
    line: int
    command: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.message}: {self.command}"


def _shell_fenced_blocks(text: str) -> Iterator[tuple[int, tuple[str, ...]]]:
    """Yield shell-fenced Markdown blocks as ``(first_line, lines)`` pairs."""

    fence_character: str | None = None
    fence_length = 0
    block_start = 0
    block_lines: list[str] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if fence_character is None:
            match = re.match(r"^(`{3,}|~{3,})\s*([^\s`]*)", stripped)
            if match is None:
                continue
            language = match.group(2).lower()
            if language not in SHELL_FENCE_LANGUAGES:
                continue
            marker = match.group(1)
            fence_character = marker[0]
            fence_length = len(marker)
            block_start = line_number + 1
            block_lines = []
            continue

        closing = fence_character * fence_length
        if stripped.startswith(closing):
            yield block_start, tuple(block_lines)
            fence_character = None
            fence_length = 0
            block_start = 0
            block_lines = []
            continue
        block_lines.append(line)

    if fence_character is not None:
        yield block_start, tuple(block_lines)


def _logical_shell_commands(
    block_start: int,
    lines: Sequence[str],
) -> Iterator[tuple[int, str]]:
    """Join backslash continuations and return executable shell commands."""

    command_parts: list[str] = []
    command_start = block_start

    for offset, raw_line in enumerate(lines):
        line_number = block_start + offset
        stripped = raw_line.strip()
        if stripped.startswith("$ "):
            stripped = stripped[2:].lstrip()

        if not command_parts and (not stripped or stripped.startswith("#")):
            continue
        if not command_parts:
            command_start = line_number

        continued = stripped.endswith("\\")
        if continued:
            stripped = stripped[:-1].rstrip()
        command_parts.append(stripped)

        if continued:
            continue
        command = " ".join(part for part in command_parts if part).strip()
        if command:
            yield command_start, command
        command_parts = []

    if command_parts:
        command = " ".join(part for part in command_parts if part).strip()
        if command:
            yield command_start, command


def _command_tokens(command: str) -> tuple[str, ...]:
    try:
        tokens = list(shlex.split(command, comments=True, posix=True))
    except ValueError:
        return ()

    while tokens:
        first = tokens[0]
        if SHELL_ASSIGNMENT_PATTERN.fullmatch(first):
            tokens.pop(0)
            continue
        if first in SHELL_WRAPPERS:
            tokens.pop(0)
            if first == "env":
                while tokens and SHELL_ASSIGNMENT_PATTERN.fullmatch(tokens[0]):
                    tokens.pop(0)
            continue
        break
    return tuple(tokens)


def _validate_causal4d_command(
    path: Path,
    line: int,
    command: str,
) -> RunbookCommandIssue | None:
    tokens = _command_tokens(command)
    if not tokens:
        return None

    executable = tokens[0]
    if LEGACY_EXECUTABLE_PATTERN.fullmatch(executable):
        return RunbookCommandIssue(
            path=path,
            line=line,
            command=command,
            message=("removed historical executable; use its grouped `causal4d` route"),
        )
    if executable != PRIMARY_EXECUTABLE:
        return None

    arguments = tokens[1:]
    if not arguments:
        return RunbookCommandIssue(
            path=path,
            line=line,
            command=command,
            message="missing grouped route or option",
        )
    if arguments[0].startswith("-"):
        return None

    if arguments[0] == "commands":
        if len(arguments) < 2 or arguments[1] not in COMMAND_MANAGEMENT_ROUTES:
            return RunbookCommandIssue(
                path=path,
                line=line,
                command=command,
                message="unknown command-registry management route",
            )
        return None

    routes = sorted(
        (spec.route for spec in grouped_commands()),
        key=len,
        reverse=True,
    )
    if any(arguments[: len(route)] == route for route in routes):
        return None
    return RunbookCommandIssue(
        path=path,
        line=line,
        command=command,
        message="unknown grouped Causal4D route",
    )


def validate_markdown(
    path: Path,
    text: str,
) -> tuple[RunbookCommandIssue, ...]:
    """Validate Causal4D commands in shell-fenced blocks from one document."""

    issues: list[RunbookCommandIssue] = []
    for block_start, lines in _shell_fenced_blocks(text):
        for line, command in _logical_shell_commands(block_start, lines):
            issue = _validate_causal4d_command(path, line, command)
            if issue is not None:
                issues.append(issue)
    return tuple(issues)


def validate_paths(
    repository_root: Path = REPOSITORY_ROOT,
    paths: Iterable[Path] = ACQUISITION_RUNBOOKS,
) -> tuple[RunbookCommandIssue, ...]:
    """Validate every configured acquisition runbook below ``repository_root``."""

    issues: list[RunbookCommandIssue] = []
    for relative_path in paths:
        document_path = repository_root / relative_path
        if not document_path.is_file():
            issues.append(
                RunbookCommandIssue(
                    path=relative_path,
                    line=1,
                    command="",
                    message="configured acquisition runbook is missing",
                )
            )
            continue
        text = document_path.read_text(encoding="utf-8")
        issues.extend(validate_markdown(relative_path, text))
    return tuple(issues)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Runbook paths relative to --repository-root (defaults to the locked set).",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository checkout containing the runbooks and src/ tree.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = tuple(args.paths) if args.paths else ACQUISITION_RUNBOOKS
    issues = validate_paths(args.repository_root.resolve(), paths)
    for issue in issues:
        print(issue.render(), file=sys.stderr)
    if issues:
        print(
            f"acquisition runbook command validation failed with {len(issues)} issue(s)",
            file=sys.stderr,
        )
        return 1
    print(f"validated Causal4D commands in {len(paths)} acquisition runbook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
