from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPOSITORY_ROOT / "scripts/ci/check_acquisition_runbook_commands.py"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_acquisition_runbook_commands",
        CHECKER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


@pytest.mark.parametrize(
    "command",
    [
        "causal4d protocol real status /tmp/protocol.json /tmp/evidence",
        "causal4d protocol readiness status /opt/causal4d /data/evidence",
        "causal4d protocol freeze validate method_freeze.json /opt/causal4d",
        "causal4d calibration execution-block evaluate folds.json",
        "causal4d commands migrate causal4d-real-protocol",
        "env MODE=confirmatory causal4d protocol real validate-dataset protocol.json data",
        "causal4d --help",
    ],
)
def test_current_grouped_commands_are_accepted(command: str) -> None:
    text = f"```bash\n{command}\n```\n"
    assert CHECKER.validate_markdown(Path("runbook.md"), text) == ()


def test_removed_historical_executable_is_rejected() -> None:
    text = """```bash
causal4d-real-protocol status protocol.json evidence
```
"""
    issues = CHECKER.validate_markdown(Path("runbook.md"), text)
    assert len(issues) == 1
    assert issues[0].line == 2
    assert "removed historical executable" in issues[0].message


def test_unknown_grouped_route_is_rejected() -> None:
    text = """```console
$ causal4d protocol definitely-not-a-route --help
```
"""
    issues = CHECKER.validate_markdown(Path("runbook.md"), text)
    assert len(issues) == 1
    assert "unknown grouped Causal4D route" in issues[0].message


def test_unknown_command_management_route_is_rejected() -> None:
    text = """```sh
causal4d commands silently-rewrite-history
```
"""
    issues = CHECKER.validate_markdown(Path("runbook.md"), text)
    assert len(issues) == 1
    assert "unknown command-registry management route" in issues[0].message


def test_backslash_continuation_reports_the_first_command_line() -> None:
    text = """```bash
# Old command copied from a pre-0.5 checklist.
causal4d-real-experiment-freeze seal \\
  /opt/causal4d \\
  method_freeze.json
```
"""
    issues = CHECKER.validate_markdown(Path("runbook.md"), text)
    assert len(issues) == 1
    assert issues[0].line == 3


def test_non_causal4d_commands_are_ignored() -> None:
    text = """```bash
python -m pip install -e .
git rev-parse HEAD
```
"""
    assert CHECKER.validate_markdown(Path("runbook.md"), text) == ()


def test_acquisition_critical_runbooks_use_current_commands() -> None:
    assert CHECKER.validate_paths(REPOSITORY_ROOT) == ()
