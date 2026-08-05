"""Enforce the complete versioned Bayesian-PhysTwin import boundary."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

ALLOWED_BAYESIAN_PHYSTWIN_MODULES = frozenset(
    {
        "bayesian_phystwin.causal4d_artifacts_v1",
        "bayesian_phystwin.causal4d_artifacts_v2",
        "bayesian_phystwin.causal4d_belief_provider_v1",
        "bayesian_phystwin.causal4d_belief_provider_v2",
        "bayesian_phystwin.causal4d_graph_provider_v1",
        "bayesian_phystwin.causal4d_provider_v1",
        "bayesian_phystwin.causal4d_provider_v2",
        "bayesian_phystwin.causal4d_public_provider_v1",
    }
)


def _python_sources() -> list[Path]:
    repository_root = Path(__file__).resolve().parents[1]
    sources: list[Path] = []
    for directory in (repository_root / "src", repository_root / "scripts"):
        if directory.exists():
            sources.extend(directory.rglob("*.py"))
    return sorted(sources)


def test_causal4d_imports_bpt_only_through_versioned_provider_modules() -> None:
    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module.startswith("bayesian_phystwin"):
                    continue
                private_names = sorted(
                    alias.name for alias in node.names if alias.name.startswith("_")
                )
                if module not in ALLOWED_BAYESIAN_PHYSTWIN_MODULES or private_names:
                    violations.append(
                        f"{path}:{node.lineno}: module={module!r}: "
                        f"private_names={private_names}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("bayesian_phystwin"):
                        continue
                    if alias.name not in ALLOWED_BAYESIAN_PHYSTWIN_MODULES:
                        violations.append(
                            f"{path}:{node.lineno}: module={alias.name!r}"
                        )
    assert not violations, "unversioned Bayesian-PhysTwin imports:\n" + "\n".join(
        violations
    )


def test_every_imported_provider_name_resolves_when_bpt_is_installed() -> None:
    if importlib.util.find_spec("bayesian_phystwin") is None:
        pytest.skip("Bayesian-PhysTwin is not installed in the core-only environment")
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module_name = node.module or ""
            if module_name not in ALLOWED_BAYESIAN_PHYSTWIN_MODULES:
                continue
            module = importlib.import_module(module_name)
            for alias in node.names:
                assert alias.name != "*"
                assert hasattr(module, alias.name), (
                    f"{path}:{node.lineno}: {module_name}.{alias.name}"
                )
