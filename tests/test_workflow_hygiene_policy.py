from __future__ import annotations

from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def test_no_temporary_self_modifying_workflows_are_shipped() -> None:
    temporary = sorted(path.name for path in WORKFLOW_DIR.glob("temporary-*.yml"))
    assert temporary == []


def test_read_only_workflows_do_not_request_repository_write_access() -> None:
    for name in ("optional-integrations.yml", "self-hosted-evaluation.yml"):
        text = (WORKFLOW_DIR / name).read_text(encoding="utf-8")
        assert "contents: write" not in text
        assert "issues: write" not in text
        assert "pull-requests: write" not in text
