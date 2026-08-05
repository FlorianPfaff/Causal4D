from pathlib import Path

import pytest


def test_tag_release_requires_public_provider_integration() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    if not workflow.is_file():
        pytest.skip("GitHub workflow is not included in the source distribution")
    text = workflow.read_text(encoding="utf-8")
    assert "Pinned Bayesian-PhysTwin installed-wheel integration" in text
    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "python -m build --wheel" in text
    assert "artifact-venv/bin/python scripts/ci/run_bpt_integration_tests.py" in text
    assert (
        "needs: [quality, core, pinned-bpt, build, installed-artifact, bundles, frozen-manifest]"
        in text
    )
    assert "BPT_READ_SSH_KEY" not in text
    assert "ssh-key:" not in text
