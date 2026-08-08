from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
REGISTRY = ROOT / ".github" / "self-hosted-jobs.json"
JOB_HEADER = re.compile(r"^  (?P<job>[A-Za-z0-9_-]+):\s*$")
PINNED_ACTION = re.compile(r"[0-9a-f]{40}")


def _job_blocks(text: str) -> dict[str, str]:
    lines = text.splitlines(keepends=True)
    try:
        jobs_start = next(
            index for index, line in enumerate(lines) if line.rstrip() == "jobs:"
        )
    except StopIteration:
        return {}

    starts: list[tuple[str, int]] = []
    for index in range(jobs_start + 1, len(lines)):
        line = lines[index]
        if line and not line[0].isspace() and line.strip():
            break
        match = JOB_HEADER.match(line.rstrip("\n"))
        if match is not None:
            starts.append((match.group("job"), index))

    blocks: dict[str, str] = {}
    for position, (job, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(lines)
        blocks[job] = "".join(lines[start:end])
    return blocks


def _job_property(block: str, name: str) -> str:
    lines = block.splitlines(keepends=True)
    prefix = f"    {name}:"
    for start, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        collected = [line]
        for candidate in lines[start + 1 :]:
            if re.match(r"^    [A-Za-z0-9_-]+:\s*", candidate):
                break
            collected.append(candidate)
        return "".join(collected)
    return ""


def _uses_self_hosted_runner(block: str) -> bool:
    runs_on = _job_property(block, "runs-on")
    if "self-hosted" in runs_on:
        return True
    strategy = _job_property(block, "strategy")
    return "matrix." in runs_on and "self-hosted" in strategy


def _discover_self_hosted_jobs() -> dict[tuple[str, str], tuple[str, str]]:
    discovered: dict[tuple[str, str], tuple[str, str]] = {}
    paths = sorted((*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for job, block in _job_blocks(text).items():
            if _uses_self_hosted_runner(block):
                discovered[(path.name, job)] = (text, block)
    return discovered


def _dispatch_only_workflow(text: str) -> bool:
    prefix = text.split("permissions:", maxsplit=1)[0]
    if re.search(r"^  workflow_dispatch:\s*$", prefix, re.MULTILINE) is None:
        return False
    other_events = ("pull_request", "push", "schedule", "workflow_call")
    return all(
        re.search(rf"^  {event}:\s*$", prefix, re.MULTILINE) is None
        for event in other_events
    )


def _main_only_errors(workflow_text: str, block: str) -> list[str]:
    errors: list[str] = []
    if "github.ref == 'refs/heads/main'" not in block:
        errors.append("missing job-level main guard")
    if (
        "github.event_name == 'workflow_dispatch'" not in block
        and not _dispatch_only_workflow(workflow_text)
    ):
        errors.append("missing dispatch-only authorization")
    if "ref: ${{ github.sha }}" not in block:
        errors.append("checkout is not bound to github.sha")
    if "git rev-parse HEAD" not in block or "GITHUB_SHA" not in block:
        errors.append("exact checkout SHA is not verified")
    if "git status --porcelain" not in block:
        errors.append("clean checkout is not verified")

    checkout_count = block.count("uses: actions/checkout@")
    if block.count("persist-credentials: false") < checkout_count:
        errors.append("one or more checkouts retain credentials")

    for line in block.splitlines():
        stripped = line.strip()
        match = re.match(r"-?\s*uses:\s*(?P<target>[^#\s]+)", stripped)
        if match is None:
            continue
        target = match.group("target")
        if target.startswith("./"):
            continue
        if (
            "@" not in target
            or PINNED_ACTION.fullmatch(target.rsplit("@", 1)[1]) is None
        ):
            errors.append(f"action is not pinned by full commit SHA: {target}")

    if "${{ secrets." in block:
        errors.append("self-hosted job references a GitHub secret")
    return errors


def test_self_hosted_job_registry_is_complete_and_unique() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload["authorization_models"]) == {"main-only"}

    entries = payload["jobs"]
    keys = [(entry["workflow"], entry["job"]) for entry in entries]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)
    assert set(keys) == set(_discover_self_hosted_jobs())


def test_every_self_hosted_job_is_main_only_exact_sha_and_secret_free() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    discovered = _discover_self_hosted_jobs()

    for entry in payload["jobs"]:
        key = (entry["workflow"], entry["job"])
        workflow_text, block = discovered[key]
        assert entry["authorization_model"] == "main-only"
        assert entry["secrets_allowed"] is False
        assert "permissions:\n  contents: read\n" in workflow_text
        assert "contents: write" not in workflow_text
        assert "issues: write" not in workflow_text
        assert "pull-requests: write" not in workflow_text
        for label in entry["runner_labels"]:
            assert label in block
        assert _main_only_errors(workflow_text, block) == [], key


def test_runner_discovery_ignores_hosted_jobs_that_only_mention_self_hosted() -> None:
    block = """  contract:
    runs-on: ubuntu-latest
    steps:
      - run: python tests/test_self_hosted_workflow_policy.py
"""
    assert _uses_self_hosted_runner(block) is False


def test_main_only_validator_accepts_a_reviewed_dispatch_fixture() -> None:
    workflow = """on:
  workflow_dispatch:
permissions:
  contents: read
"""
    block = """  evaluate:
    if: >-
      github.event_name == 'workflow_dispatch' &&
      github.ref == 'refs/heads/main'
    runs-on: [self-hosted, Linux, X64]
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          ref: ${{ github.sha }}
          persist-credentials: false
      - run: |
          test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"
          test -z "$(git status --porcelain=v1)"
"""
    assert _main_only_errors(workflow, block) == []


def test_main_only_validator_rejects_an_unauthorized_or_stale_fixture() -> None:
    workflow = """on:
  workflow_dispatch:
  pull_request:
permissions:
  contents: read
"""
    block = """  evaluate:
    if: github.ref == 'refs/heads/feature'
    runs-on: [self-hosted, Linux, X64]
    steps:
      - uses: actions/checkout@v7
"""
    errors = _main_only_errors(workflow, block)
    assert "missing job-level main guard" in errors
    assert "missing dispatch-only authorization" in errors
    assert "checkout is not bound to github.sha" in errors
    assert "exact checkout SHA is not verified" in errors
    assert "clean checkout is not verified" in errors
    assert "one or more checkouts retain credentials" in errors
    assert any(error.startswith("action is not pinned") for error in errors)
