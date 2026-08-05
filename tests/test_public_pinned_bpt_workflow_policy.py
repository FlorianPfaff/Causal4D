from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "public-pinned-bpt.yml"
)


def test_public_pinned_provider_requires_no_repository_secret() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "ref: ${{ steps.pin.outputs.sha }}" in text
    assert "BPT_READ_SSH_KEY" not in text
    assert "ssh-key:" not in text
    assert "PROB4D_READ_TOKEN" not in text


def test_provider_boundary_uses_built_wheels_and_isolated_imports() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("python -m build --wheel") == 2
    assert "public-pinned-bpt-venv" in text
    assert "env -u PYTHONPATH" in text
    assert "Reject checkout-resolved imports" in text
    assert "origin.is_relative_to(root)" in text
    assert "--import-mode=importlib" in text
    assert "CAUSAL4D_REQUIRE_BPT_PROVIDER" in text


def test_provider_revision_and_wheel_bytes_are_recorded() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "read_bpt_pin.py" in text
    assert "git -C causal4d rev-parse HEAD" in text
    assert "git -C bayesian-phystwin rev-parse HEAD" in text
    assert "sha256sum ./*.whl | sort" in text
    assert "public-pinned-bpt-wheel-sha256.txt" in text
    assert "Upload wheel identities" in text


def test_public_gate_runs_continuously_and_on_schedule() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "push:" in text
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert "cancel-in-progress: true" in text
