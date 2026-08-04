from __future__ import annotations

from pathlib import Path


_WORKFLOW_DIRECTORY = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows"
)
_FORBIDDEN_NAME_FRAGMENTS = (
    "apply-contact",
    "fix-contact",
    "format-contact",
    "repair-contact",
    "bootstrap-contact",
    "contact-bootstrap",
)
_BRANCH_NAME = "agent/reproducibility-contact-diagnostics"


def _contact_workflows() -> list[Path]:
    return sorted(
        path
        for pattern in ("*.yml", "*.yaml")
        for path in _WORKFLOW_DIRECTORY.glob(pattern)
        if "contact" in path.name.lower()
    )


def test_contact_workflow_names_exclude_branch_repair_helpers() -> None:
    workflows = _contact_workflows()
    assert workflows, "the permanent contact diagnostic workflow is missing"
    for path in workflows:
        lowered = path.name.lower()
        assert all(fragment not in lowered for fragment in _FORBIDDEN_NAME_FRAGMENTS)


def test_contact_workflows_are_read_only_and_branch_agnostic() -> None:
    for path in _contact_workflows():
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "contents: write" not in lowered
        assert "pull-requests: write" not in lowered
        assert "git commit" not in lowered
        assert "git push" not in lowered
        assert _BRANCH_NAME not in text


def test_provenance_contract_changes_trigger_the_diagnostic_workflow() -> None:
    workflow = _WORKFLOW_DIRECTORY / "contact-posterior-diagnostics.yml"
    text = workflow.read_text(encoding="utf-8")
    assert '"src/causal4d/contact_posterior_provenance.py"' in text
    assert '"tests/test_contact_posterior_provenance.py"' in text
    assert '"tests/test_contact_workflow_inventory_policy.py"' in text
