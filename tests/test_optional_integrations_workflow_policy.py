from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "optional-integrations.yml"
)


def test_optional_provider_jobs_use_public_immutable_checkout() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("repository: IPS-Stuttgart/BayesianPhysTwin") == 3
    assert text.count("ref: ${{ steps.pin.outputs.sha }}") == 3
    assert text.count("Check out pinned public BayesianPhysTwin") == 3
    assert "BPT_READ_SSH_KEY" not in text
    assert "ssh-key:" not in text
    assert "steps.access.outputs.enabled" not in text
    assert "Detect private-repository deploy key" not in text
    assert "Require private-repository deploy key" not in text


def test_optional_integrations_do_not_use_editable_provider_installs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m pip install -e" not in text
    assert 'python -m pip install "./_bpt[dev]" opencv-python' in text
    assert 'python -m pip install ".[dev,warp]" ./_bpt' in text
    assert 'python -m pip install ".[dev,vision,warp]" ./_bpt' in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "inputs.run_gpu" in text
