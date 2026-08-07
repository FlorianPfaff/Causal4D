"""Policy checks for the permanent exact-head validation workflow."""

from pathlib import Path


WORKFLOW = Path(".github/workflows/exact-head-validation.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_exact_head_validation_has_manual_and_scheduled_queue() -> None:
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert '<!-- exact-head-validation: queued -->' in text
    assert "same-repository PRs only" in text
    assert 'pull["base"]["ref"] != "main"' in text
    assert 'WORKFLOW_REF"] != "refs/heads/main"' in text


def test_exact_head_checkout_is_immutable_and_uncredentialed() -> None:
    text = _workflow_text()
    assert "ref: ${{ needs.select.outputs.head_sha }}" in text
    assert text.count("fetch-depth: 0") >= 2
    assert text.count("persist-credentials: false") >= 3
    assert text.count("git rev-parse HEAD") >= 2
    assert text.count("pull-request head changed during exact-head validation") >= 2
    assert "git merge-tree --write-tree" in text
    assert "pull_request_target" not in text
    assert "contents: write" not in text


def test_exact_head_validation_matches_authoritative_stack() -> None:
    text = _workflow_text()
    for version in ('"3.10"', '"3.12"', '"3.14"'):
        assert version in text
    for command in (
        "python -W error::SyntaxWarning -m compileall",
        "python -m pytest",
        "python -m ruff check .",
        "python -m ruff format --check",
        "python -m mypy --python-version 3.12",
        "python -m build",
        "python -m twine check dist/*",
        "python scripts/ci/read_bpt_pin.py",
        "scripts/ci/run_bpt_integration_tests.py",
    ):
        assert command in text


def test_exact_head_result_is_bound_and_one_shot() -> None:
    text = _workflow_text()
    assert '"artifact_kind": "Causal4DExactHeadValidation"' in text
    assert '"head_sha": expected_head' in text
    assert '"base_sha": os.environ["EXPECTED_BASE_SHA"]' in text
    assert '"head_still_current": head_current' in text
    assert "exact-head-validation.json" in text
    assert "body.replace(marker, completion, 1)" in text
