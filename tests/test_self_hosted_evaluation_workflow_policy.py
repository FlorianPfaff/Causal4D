from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "self-hosted-evaluation.yml"
)


def test_requested_provider_validation_uses_ssh_and_fails_closed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Require private BayesianPhysTwin credential" in text
    assert "BPT_READ_SSH_KEY: ${{ secrets.BPT_READ_SSH_KEY }}" in text
    assert "ssh-key: ${{ secrets.BPT_READ_SSH_KEY }}" in text
    assert "provider checks were requested and cannot run" in text
    assert text.count("if: ${{ inputs.run_bpt }}") >= 5
    assert "BPT_READ_TOKEN" not in text
    assert "continue-on-error: true" not in text
    assert "Report unavailable Bayesian-PhysTwin access" not in text


def test_self_hosted_gpu_stack_does_not_use_actions_pip_cache() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "cache: pip" not in text
    assert 'python -m pip install -e ".[dev,warp]"' in text
