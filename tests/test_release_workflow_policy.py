from pathlib import Path

import pytest


def test_tag_release_requires_private_provider_integration() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    if not workflow.is_file():
        pytest.skip("GitHub workflow is not included in the source distribution")
    text = workflow.read_text(encoding="utf-8")
    assert "Require private-provider integration for releases" in text
    assert "startsWith(github.ref, 'refs/tags/v')" in text
    assert "steps.access.outputs.enabled != 'true'" in text
    assert "Release tags require BPT_READ_TOKEN" in text
