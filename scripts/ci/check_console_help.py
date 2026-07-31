#!/usr/bin/env python3
"""Verify installed package metadata and every Causal4D command's ``--help``."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib


def _declared_scripts(pyproject: Path) -> dict[str, str]:
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    scripts = project.get("scripts", {})
    if not isinstance(scripts, dict) or not scripts:
        raise ValueError("pyproject.toml does not declare any console scripts")
    return {str(name): str(target) for name, target in scripts.items()}


def _installed_scripts(distribution: str) -> dict[str, str]:
    installed = importlib.metadata.distribution(distribution)
    return {
        entry_point.name: entry_point.value
        for entry_point in installed.entry_points
        if entry_point.group == "console_scripts"
        and (entry_point.name == "causal4d" or entry_point.name.startswith("causal4d-"))
    }


def _require_installed_file(distribution: str, relative_path: str) -> Path:
    installed = importlib.metadata.distribution(distribution)
    files = installed.files
    if files is None:
        raise RuntimeError(
            f"installed distribution {distribution!r} exposes no file inventory"
        )
    normalized = {
        str(package_path).replace("\\", "/"): package_path
        for package_path in files
    }
    package_path = normalized.get(relative_path)
    if package_path is None:
        raise RuntimeError(
            f"installed distribution {distribution!r} does not contain {relative_path}"
        )
    resolved = Path(installed.locate_file(package_path))
    if not resolved.is_file():
        raise RuntimeError(
            f"installed distribution records {relative_path}, but {resolved} is absent"
        )
    return resolved


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def verify_console_help(
    pyproject: Path,
    *,
    distribution: str = "causal4d",
    timeout_seconds: float = 30.0,
) -> None:
    typing_marker = _require_installed_file(distribution, "causal4d/py.typed")
    print(f"verified installed PEP 561 marker: {typing_marker}")

    declared = _declared_scripts(pyproject)
    installed = _installed_scripts(distribution)
    if installed != declared:
        missing = sorted(set(declared) - set(installed))
        unexpected = sorted(set(installed) - set(declared))
        mismatched = sorted(
            name
            for name in set(declared) & set(installed)
            if declared[name] != installed[name]
        )
        raise RuntimeError(
            "installed console-script metadata differs from pyproject.toml: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    failures: list[str] = []
    environment = _clean_environment()
    with tempfile.TemporaryDirectory(prefix="causal4d-cli-help-") as directory:
        for name in sorted(declared):
            executable = shutil.which(name)
            if executable is None:
                failures.append(f"{name}: executable is not on PATH")
                continue
            try:
                completed = subprocess.run(
                    [executable, "--help"],
                    cwd=directory,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{name}: timed out after {timeout_seconds:g} s")
                continue
            output = completed.stdout + completed.stderr
            if completed.returncode != 0:
                failures.append(
                    f"{name}: exited with {completed.returncode}\n{output[-4000:]}"
                )
                continue
            if "usage:" not in output.lower():
                failures.append(
                    f"{name}: successful help output did not contain 'usage:'"
                )
                continue
            print(f"ok {name}")

    if failures:
        joined = "\n\n".join(failures)
        raise RuntimeError(
            f"{len(failures)} of {len(declared)} console scripts failed --help:\n{joined}"
        )
    print(f"verified --help for all {len(declared)} installed console scripts")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--distribution", default="causal4d")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    verify_console_help(
        arguments.pyproject,
        distribution=arguments.distribution,
        timeout_seconds=arguments.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
