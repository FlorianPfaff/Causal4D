from __future__ import annotations

import ast
from pathlib import Path

from causal4d.cli.command_registry import find_command, grouped_commands


def _command_source(target: str) -> Path:
    module_name, separator, function_name = target.partition(":")
    assert separator == ":" and function_name
    repository_root = Path(__file__).resolve().parents[1]
    source = repository_root / "src" / Path(*module_name.split("."))
    return source.with_suffix(".py")


def _unsafe_deserialization_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pickle_modules: set[str] = set()
    pickle_loads: set[str] = set()
    numpy_modules: set[str] = set()
    numpy_loads: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name == "pickle":
                    pickle_modules.add(local_name)
                elif alias.name == "numpy":
                    numpy_modules.add(local_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "pickle":
                for alias in node.names:
                    if alias.name in {"load", "loads"}:
                        pickle_loads.add(alias.asname or alias.name)
            elif node.module == "numpy":
                for alias in node.names:
                    if alias.name == "load":
                        numpy_loads.add(alias.asname or alias.name)

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in pickle_loads:
            violations.append(f"{path}:{node.lineno}: direct pickle deserialization")
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in pickle_modules
            and node.func.attr in {"load", "loads"}
        ):
            violations.append(f"{path}:{node.lineno}: direct pickle deserialization")

        numpy_load = (
            isinstance(node.func, ast.Name) and node.func.id in numpy_loads
        ) or (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in numpy_modules
            and node.func.attr == "load"
        )
        if numpy_load:
            for keyword in node.keywords:
                if (
                    keyword.arg == "allow_pickle"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    violations.append(
                        f"{path}:{node.lineno}: numpy.load(allow_pickle=True)"
                    )
    return violations


def test_claim_bearing_command_modules_do_not_deserialize_pickle_directly() -> None:
    violations: list[str] = []
    for command in grouped_commands():
        if not command.claim_bearing:
            continue
        source = _command_source(command.target)
        assert source.is_file(), command.target
        violations.extend(_unsafe_deserialization_calls(source))
    assert not violations, "unsafe claim-bearing input paths:\n" + "\n".join(
        violations
    )


def test_physical_evaluator_uses_safe_target_and_atomic_publication() -> None:
    command = find_command("evidence/physical-counterfactual/evaluate")
    assert command.claim_bearing
    source = _command_source(command.target)
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    imported: set[tuple[str | None, str]] = set()
    direct_writes: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update((node.module, alias.name) for alias in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"write_text", "write_bytes"}
        ):
            direct_writes.append(node.lineno)

    assert (
        "causal4d.held_out_target",
        "load_held_out_physical_target",
    ) in imported
    assert (
        "causal4d.physical_evaluation_record",
        "save_physical_counterfactual_evaluation_record",
    ) in imported
    assert not direct_writes, f"direct evidence writes at lines {direct_writes}"
