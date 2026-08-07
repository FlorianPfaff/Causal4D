from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "decision-trace-installed-wheel.yml"
BPT_PIN = (
    (ROOT / "requirements" / "ci" / "bayesian-phystwin-three-repository.sha")
    .read_text(encoding="utf-8")
    .strip()
)
PROB4D_PIN = (
    (ROOT / "requirements" / "ci" / "prob4d-three-repository.sha")
    .read_text(encoding="utf-8")
    .strip()
)
REQUIRED_TRIGGER_PATHS = (
    ".github/workflows/decision-trace-installed-wheel.yml",
    "docs/decision_trace.md",
    "pyproject.toml",
    "requirements/ci/bayesian-phystwin-three-repository.sha",
    "requirements/ci/prob4d-three-repository.sha",
    "src/causal4d/__init__.py",
    "src/causal4d/atomic_io.py",
    "src/causal4d/cli/root.py",
    "src/causal4d/cli/stack.py",
    "src/causal4d/decision_trace.py",
    "src/causal4d/immutable_json.py",
    "src/causal4d/stack_lock.py",
    "tests/test_decision_trace.py",
    "tests/test_decision_trace_workflow_policy.py",
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _preamble(text: str) -> str:
    return text.split("\njobs:\n", maxsplit=1)[0]


def test_decision_trace_changes_trigger_pull_request_and_main_validation() -> None:
    text = _workflow_text()
    preamble = _preamble(text)

    assert "pull_request:" in preamble
    assert "push:" in preamble
    assert "branches: [main]" in preamble
    assert "workflow_dispatch:" in preamble
    for path in REQUIRED_TRIGGER_PATHS:
        assert text.count(f'- "{path}"') == 2


def test_decision_trace_workflow_is_unprivileged_and_hosted() -> None:
    text = _workflow_text()
    preamble = _preamble(text)

    assert "permissions:\n  contents: read" in preamble
    assert "pull_request_target:" not in preamble
    assert "workflow_run:" not in preamble
    assert "issue_comment:" not in preamble
    assert "security-events: write" not in text
    assert "contents: write" not in text
    assert "self-hosted" not in text
    assert "secrets." not in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on:" in text
    assert text.count("runs-on:") == 1


def test_companion_revisions_are_exact_checked_in_pins() -> None:
    text = _workflow_text()

    assert f"ref: {BPT_PIN}" in text
    assert f"ref: {PROB4D_PIN}" in text
    assert "ref: main" not in text
    assert "github.event.pull_request.head" not in text
    assert "github.head_ref" not in text
    assert "workflow_dispatch.inputs" not in text
    assert (
        "cat causal4d/requirements/ci/bayesian-phystwin-three-repository.sha"
        in text
    )
    assert "cat causal4d/requirements/ci/prob4d-three-repository.sha" in text


def test_all_actions_and_checkouts_are_immutable_and_read_only() -> None:
    text = _workflow_text()

    checkout = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    setup_python = (
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
    )
    assert text.count(checkout) == 3
    assert text.count(setup_python) == 1
    assert text.count("persist-credentials: false") == 3
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        reference = stripped.rsplit("@", maxsplit=1)[1].split()[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)


def test_contracts_run_from_the_installed_stack_not_source_trees() -> None:
    text = _workflow_text()

    assert "python -m build --wheel" in text
    assert '"$venv/bin/python" -m pip install' in text
    assert '"$venv/bin/python" -m pip check' in text
    assert '"$venv/bin/causal4d" stack create' in text
    assert '"$venv/bin/causal4d" stack verify' in text
    assert "from causal4d import decision_trace" in text
    assert "source-tree import detected" in text
    assert 'run_dir="$RUNNER_TEMP/decision-trace-installed-wheel-tests"' in text
    assert "cp causal4d/tests/test_decision_trace.py" in text
    assert "env -u PYTHONPATH" in text
    assert "--import-mode=importlib" in text
