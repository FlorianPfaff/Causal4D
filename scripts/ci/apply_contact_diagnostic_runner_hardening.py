#!/usr/bin/env python3
"""Apply the reviewed contact-diagnostic runner-boundary cleanup."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "contact-posterior-diagnostics.yml"
POLICY = ROOT / "tests" / "test_contact_posterior_workflow_policy.py"

OLD_JOB_IF = """    if: >-
      github.event_name == 'workflow_dispatch' ||
      (
        github.event.pull_request.head.repo.full_name == github.repository &&
        github.event.pull_request.head.ref !=
        'ci/pr193-latent-contact-stability-main-v2'
      )
"""
NEW_JOB_IF = """    if: >-
      (github.event_name == 'pull_request' &&
       github.event.pull_request.head.repo.full_name == github.repository) ||
      (github.event_name == 'workflow_dispatch' &&
       (inputs.runner != 'self-hosted' ||
        github.ref == 'refs/heads/main'))
"""

OLD_CHECKOUT = """      - name: Check out Causal4D
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          fetch-depth: 0
          persist-credentials: false
          clean: true

"""
NEW_CHECKOUT = """      - name: Check out exact Causal4D revision
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          fetch-depth: 0
          persist-credentials: false
          clean: true

      - name: Verify exact clean revision
        env:
          EXPECTED_SHA: ${{ github.event.pull_request.head.sha || github.sha }}
        shell: bash
        run: |
          set -euo pipefail
          test "$(git rev-parse HEAD)" = "${EXPECTED_SHA}"
          test -z "$(git status --porcelain=v1)"

"""

POLICY_APPEND = '''


def test_contact_diagnostic_self_hosted_dispatch_is_main_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "github.event_name == 'workflow_dispatch'" in text
    assert "inputs.runner != 'self-hosted'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in text
    assert "EXPECTED_SHA: ${{ github.event.pull_request.head.sha || github.sha }}" in text
    assert 'test "$(git rev-parse HEAD)" = "${EXPECTED_SHA}"' in text


def test_completed_pr193_self_hosted_job_is_not_live() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "latent-contact-stability:" not in text
    assert "ci/pr193-latent-contact-stability-main-v2" not in text
    assert 'DIAGNOSTIC_SEEDS: "3000:3500"' not in text
'''


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name}: expected one reviewed block, found {count}")
    return text.replace(old, new)


def main() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    workflow = _replace_once(workflow, OLD_JOB_IF, NEW_JOB_IF, name="job guard")
    workflow = _replace_once(workflow, OLD_CHECKOUT, NEW_CHECKOUT, name="checkout")

    stale_marker = "\n  latent-contact-stability:\n"
    if workflow.count(stale_marker) != 1:
        raise SystemExit("expected one completed latent-contact-stability job")
    workflow = workflow.split(stale_marker, maxsplit=1)[0].rstrip() + "\n"
    WORKFLOW.write_text(workflow, encoding="utf-8")

    policy = POLICY.read_text(encoding="utf-8")
    if "test_contact_diagnostic_self_hosted_dispatch_is_main_only" in policy:
        raise SystemExit("policy additions are already present")
    POLICY.write_text(policy.rstrip() + POLICY_APPEND + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
