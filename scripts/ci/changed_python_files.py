#!/usr/bin/env python3
"""List changed tracked Python files, falling back to the full tracked set."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _git(*arguments: str) -> list[str]:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def changed_python_files(base: str | None, head: str) -> tuple[Path, ...]:
    invalid_base = not base or set(base) == {"0"}
    if not invalid_base:
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{base}^{{commit}}"],
            check=False,
            capture_output=True,
        )
        invalid_base = probe.returncode != 0
    if invalid_base:
        names = _git("ls-files", "*.py")
    else:
        assert base is not None
        names = _git(
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            base,
            head,
            "--",
            "*.py",
        )
    return tuple(Path(name) for name in names if Path(name).is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base")
    parser.add_argument("--head", default="HEAD")
    arguments = parser.parse_args(argv)
    for path in changed_python_files(arguments.base, arguments.head):
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
