from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "bayesian-phystwin-provider-compatibility.yml"
)


def test_trusted_events_cannot_pass_by_skipping_private_repositories() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "credential-gate:" in text
    assert "Require private-repository read access" in text
    assert "exit 1" in text
    assert "needs: credential-gate" in text
    assert "steps.access.outputs.enabled" not in text
    assert "the required three-repository golden path cannot run" in text


def test_external_fork_limitation_is_separate_and_explicit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "external-pull-request:" in text
    assert "External PR cannot access private providers" in text
    assert "github.event.pull_request.head.repo.full_name != github.repository" in text
    assert "maintainers must run the installed-wheel golden path" in text
