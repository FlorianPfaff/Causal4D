from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / ".github" / "workflows"
_USES_LINE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*(?P<value>[^#\s]+)\s*(?:#\s*(?P<comment>.+))?$"
)
_FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERSION_COMMENT = re.compile(r"^v[0-9]+(?:\.[0-9]+){0,2}(?:[-+][A-Za-z0-9.-]+)?$")


def _pin_errors(text: str, *, source: str) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _USES_LINE.match(line)
        if match is None:
            continue
        value = match.group("value").strip("\"'")
        if value.startswith(("./", "docker://")):
            continue
        if "@" not in value:
            errors.append(
                f"{source}:{line_number}: external action has no ref: {value}"
            )
            continue
        action, ref = value.rsplit("@", 1)
        if "/" not in action or not _FULL_COMMIT.fullmatch(ref):
            errors.append(
                f"{source}:{line_number}: external action must use a full "
                f"lowercase commit SHA: {value}"
            )
        comment = (match.group("comment") or "").strip().split(maxsplit=1)[0:1]
        if not comment or _VERSION_COMMENT.fullmatch(comment[0]) is None:
            errors.append(
                f"{source}:{line_number}: immutable action pin needs a version "
                f"annotation such as '# v7': {value}"
            )
    return errors


def test_every_external_github_action_is_immutably_pinned() -> None:
    workflow_files = sorted(WORKFLOW_ROOT.glob("*.yml")) + sorted(
        WORKFLOW_ROOT.glob("*.yaml")
    )
    assert workflow_files, "no GitHub Actions workflows found"

    errors: list[str] = []
    for path in workflow_files:
        errors.extend(
            _pin_errors(
                path.read_text(encoding="utf-8"),
                source=path.relative_to(ROOT).as_posix(),
            )
        )
    assert not errors, "\n".join(errors)


def test_pin_policy_rejects_tags_branches_and_short_shas() -> None:
    text = "\n".join(
        (
            "- uses: actions/checkout@v7",
            "- uses: owner/action@main",
            "- uses: owner/action@0123456789abcdef # v1",
        )
    )
    errors = _pin_errors(text, source="fixture.yml")

    assert len(errors) == 5
    assert all("fixture.yml" in error for error in errors)


def test_pin_policy_allows_local_docker_and_annotated_commit_uses() -> None:
    pinned = "0123456789abcdef0123456789abcdef01234567"
    text = "\n".join(
        (
            "- uses: ./local-action",
            "- uses: docker://python:3.12",
            f"- uses: owner/action@{pinned} # v1.2.3",
        )
    )

    assert _pin_errors(text, source="fixture.yml") == []
