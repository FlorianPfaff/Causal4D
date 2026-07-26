"""Prevent Causal4D from depending on Bayesian-PhysTwin private names again."""

from __future__ import annotations

import ast
from pathlib import Path


def _python_sources() -> list[Path]:
    repository_root = Path(__file__).resolve().parents[1]
    sources = []
    for directory in (repository_root / "src", repository_root / "scripts"):
        if directory.exists():
            sources.extend(directory.rglob("*.py"))
    return sorted(sources)


def test_causal4d_imports_only_public_bayesian_phystwin_modules_and_names() -> None:
    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module.startswith("bayesian_phystwin"):
                    continue
                private_module = any(
                    component.startswith("_") for component in module.split(".")
                )
                private_names = [
                    alias.name for alias in node.names if alias.name.startswith("_")
                ]
                if private_module or private_names:
                    violations.append(
                        f"{path}:{node.lineno}: {module}: {private_names}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("bayesian_phystwin"):
                        continue
                    if any(
                        component.startswith("_")
                        for component in alias.name.split(".")
                    ):
                        violations.append(
                            f"{path}:{node.lineno}: {alias.name}"
                        )
    assert not violations, "private Bayesian-PhysTwin imports:\n" + "\n".join(
        violations
    )
