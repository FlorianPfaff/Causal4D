"""Keep BPT graph and controller geometry behind the versioned provider."""

from __future__ import annotations

import ast
from pathlib import Path


_BLOCKED_MODULES = {
    "bayesian_phystwin.phystwin_graph",
    "bayesian_phystwin.phystwin_controller_sensitivity",
}


def _production_python_sources() -> list[Path]:
    repository_root = Path(__file__).resolve().parents[1]
    sources: list[Path] = []
    for directory in (repository_root / "src", repository_root / "scripts"):
        if directory.exists():
            sources.extend(directory.rglob("*.py"))
    return sorted(sources)


def test_graph_and_controller_imports_use_versioned_provider() -> None:
    violations: list[str] = []
    for path in _production_python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _BLOCKED_MODULES:
                violations.append(f"{path}:{node.lineno}: {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _BLOCKED_MODULES:
                        violations.append(f"{path}:{node.lineno}: {alias.name}")
    assert not violations, (
        "direct Bayesian-PhysTwin graph/controller imports:\n" + "\n".join(violations)
    )
