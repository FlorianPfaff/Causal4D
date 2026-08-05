from __future__ import annotations

import ast
from pathlib import Path

from causal4d.cli.command_registry import find_command, grouped_commands


def _module_path(module_name: str) -> Path:
    repository_root = Path(__file__).resolve().parents[1]
    relative = Path(*module_name.split(".")).with_suffix(".py")
    for source_root in (repository_root / "src", repository_root):
        candidate = source_root / relative
        if candidate.is_file():
            return candidate
    raise AssertionError(f"cannot resolve command module {module_name!r}")


def test_claim_bearing_commands_do_not_import_pickle_directly() -> None:
    violations: list[str] = []
    for command in grouped_commands():
        if not command.claim_bearing:
            continue
        module_name = command.target.split(":", 1)[0]
        path = _module_path(module_name)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pickle":
                        violations.append(f"{path}:{node.lineno}: import pickle")
            elif isinstance(node, ast.ImportFrom) and node.module == "pickle":
                violations.append(f"{path}:{node.lineno}: from pickle import ...")
    message = "claim-bearing commands import pickle directly:\n" + "\n".join(
        violations
    )
    assert not violations, message


def test_physical_target_routes_have_the_intended_claim_boundary() -> None:
    importer = find_command("evidence/physical-target/import-legacy")
    evaluator = find_command("evidence/physical-counterfactual/evaluate")
    assert importer.claim_bearing
    assert importer.extras == ("phystwin",)
    assert evaluator.claim_bearing
    assert evaluator.extras == ()
    assert evaluator.requires == ()


def test_physical_evaluator_has_no_bayesian_phystwin_imports() -> None:
    evaluator = find_command("evidence/physical-counterfactual/evaluate")
    module_name = evaluator.target.split(":", 1)[0]
    path = _module_path(module_name)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    provider_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            provider_imports.extend(
                alias.name
                for alias in node.names
                if alias.name.startswith("bayesian_phystwin")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("bayesian_phystwin"):
                provider_imports.append(module)
    assert not provider_imports, (
        "physical evaluator imports an optional BayesianPhysTwin provider: "
        f"{provider_imports}"
    )
