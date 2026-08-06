from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "security-scanning.yml"


def test_security_scanning_has_read_only_triggers_and_permissions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "push:" in text
    assert "pull_request:" in text
    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "persist-credentials: false" in text
    assert "continue-on-error: true" not in text


def test_codeql_scans_python_and_workflow_sources() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "language: [python, actions]" in text
    assert "build-mode: none" in text
    assert "queries: security-extended" in text
    assert "security-events: write" in text
    assert "github/codeql-action/init@" in text
    assert "github/codeql-action/analyze@" in text
    assert 'category: "/language:${{ matrix.language }}"' in text


def test_dependency_audit_is_strict_pinned_and_archived() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'python -m pip install "pip-audit==2.10.1"' in text
    assert "python -m pip_audit" in text
    assert "--strict" in text
    assert "--progress-spinner off" in text
    assert "--output pip-audit.json" in text
    assert "if: always()" in text
    assert "actions/upload-artifact@" in text
    assert "if-no-files-found: warn" in text
