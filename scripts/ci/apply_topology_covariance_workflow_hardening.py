#!/usr/bin/env python3
"""Restrict the sealed topology-covariance reproduction to reviewed main."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "contact-topology-covariance.yml"

OLD_POLICY_SHA = "66cc1443b8f99563f99744c46584df13b1b52b5e0828c3b99cab5c96028f2e66"
NEW_POLICY_SHA = "04481f7fd2b2965d3439dad9f1b9ca3068f6b5dea1bc62674383571c73667ae1"


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name}: expected one reviewed block, found {count}")
    return text.replace(old, new)


def main() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        """  pull_request:
    branches: [main]
    paths:
      - ".github/workflows/contact-topology-covariance.yml"
      - "docs/contact_topology_covariance.md"
      - "scripts/ci/run_contact_topology_covariance_diagnostic.py"
      - "src/causal4d/contact_topology_covariance_diagnostic.py"
      - "tests/test_contact_topology_covariance_diagnostic.py"
      - "tests/test_contact_topology_covariance_workflow_policy.py"
""",
        "",
        name="pull-request trigger",
    )
    text = _replace_once(
        text,
        """    if: >-
      github.event_name == 'workflow_dispatch' ||
      github.event.pull_request.head.repo.full_name == github.repository
""",
        """    if: >-
      github.event_name == 'workflow_dispatch' &&
      github.ref == 'refs/heads/main'
""",
        name="main-only job guard",
    )
    text = _replace_once(
        text,
        """      - name: Check out Causal4D
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          fetch-depth: 0
          persist-credentials: false
          clean: true

""",
        """      - name: Check out exact reviewed Causal4D revision
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ github.sha }}
          fetch-depth: 0
          persist-credentials: false
          clean: true

      - name: Verify reviewed main revision
        shell: bash
        run: |
          set -euo pipefail
          test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"
          test -z "$(git status --porcelain=v1)"

""",
        name="exact checkout",
    )
    text = _replace_once(
        text,
        """          python -m pip install -e ".[dev]"
""",
        """          python -m pip install ".[dev]"
""",
        name="non-editable install",
    )
    text = _replace_once(
        text,
        f"{OLD_POLICY_SHA}  tests/test_contact_topology_covariance_workflow_policy.py",
        f"{NEW_POLICY_SHA}  tests/test_contact_topology_covariance_workflow_policy.py",
        name="policy content lock",
    )
    WORKFLOW.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
