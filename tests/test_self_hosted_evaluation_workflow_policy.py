from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "self-hosted-evaluation.yml"
)


def test_requested_provider_validation_uses_public_pinned_wheel() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "ref: ${{ steps.pin.outputs.sha }}" in text
    assert "Check out pinned public BayesianPhysTwin" in text
    assert "Build and install exact BayesianPhysTwin wheel" in text
    assert "bayesian_phystwin-*.whl" in text
    assert "self-hosted-evaluation/wheel-sha256.txt" in text
    assert text.count("if: ${{ inputs.run_bpt }}") >= 4
    assert "BPT_READ_SSH_KEY" not in text
    assert "ssh-key:" not in text
    assert "continue-on-error: true" not in text


def test_self_hosted_gpu_stack_uses_installed_wheels_without_actions_cache() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "cache: pip" not in text
    assert "Build and install exact Causal4D wheel" in text
    assert "python -m build --wheel --outdir self-hosted-evaluation/wheels ." in text
    assert "Causal4D resolved from the checkout instead of the wheel" in text
    assert "BayesianPhysTwin resolved from the checkout instead of the wheel" in text
    assert "python -m pip install -e" not in text
