#!/usr/bin/env python3
"""Apply main-only exact-SHA guards to remaining self-hosted workflows."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILAMENT = ROOT / ".github" / "workflows" / "deform360-filament-support.yml"
CONCENTRATION = (
    ROOT / ".github" / "workflows" / "contact-posterior-concentration.yml"
)
DEMO = ROOT / ".github" / "workflows" / "controlled-demo-video.yml"


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name}: expected one reviewed block, found {count}")
    return text.replace(old, new)


def _harden_filament() -> None:
    text = FILAMENT.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        """    if: >-
      (github.event_name == 'workflow_dispatch' && inputs.run_source_diagnostic) ||
      (github.event_name == 'pull_request' &&
       github.event.pull_request.head.repo.full_name == github.repository)
""",
        """    if: >-
      github.event_name == 'workflow_dispatch' &&
      github.ref == 'refs/heads/main' &&
      inputs.run_source_diagnostic
""",
        name="filament source-job guard",
    )
    text = _replace_once(
        text,
        """      - name: Check out Causal4D with parent history
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
          clean: true
          fetch-depth: 0

""",
        """      - name: Check out exact reviewed Causal4D revision
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ github.sha }}
          persist-credentials: false
          clean: true
          fetch-depth: 0

      - name: Verify reviewed main revision
        shell: bash
        run: |
          set -euo pipefail
          test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"
          test -z "$(git status --porcelain=v1)"

""",
        name="filament checkout",
    )
    FILAMENT.write_text(text, encoding="utf-8")


def _harden_concentration() -> None:
    text = CONCENTRATION.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        """      - "tests/test_contact_concentration_diagnostic.py"
""",
        """      - "tests/test_contact_concentration_diagnostic.py"
      - "tests/test_contact_concentration_workflow_policy.py"
""",
        name="concentration trigger coverage",
    )
    text = _replace_once(
        text,
        """    if: >-
      github.event_name == 'workflow_dispatch' ||
      github.event.pull_request.head.repo.full_name == github.repository
""",
        """    if: >-
      (github.event_name == 'pull_request' &&
       github.event.pull_request.head.repo.full_name == github.repository) ||
      (github.event_name == 'workflow_dispatch' &&
       (inputs.runner != 'self-hosted' ||
        github.ref == 'refs/heads/main'))
""",
        name="concentration runner guard",
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
        """      - name: Check out exact Causal4D revision
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

""",
        name="concentration checkout",
    )
    text = _replace_once(
        text,
        """.contact-concentration-venv/bin/python -m pip install -e ".[dev]"
""",
        """.contact-concentration-venv/bin/python -m pip install ".[dev]"
""",
        name="concentration non-editable install",
    )
    CONCENTRATION.write_text(text, encoding="utf-8")


def _harden_demo() -> None:
    text = DEMO.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        """  render:
    name: Render controlled MP4, GIF, and poster
    runs-on: >-
""",
        """  render:
    name: Render controlled MP4, GIF, and poster
    if: >-
      inputs.runner != 'self-hosted' ||
      github.ref == 'refs/heads/main'
    runs-on: >-
""",
        name="controlled-demo runner guard",
    )
    text = _replace_once(
        text,
        """      - name: Check out Causal4D
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false

""",
        """      - name: Check out exact reviewed Causal4D revision
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ github.sha }}
          fetch-depth: 1
          persist-credentials: false
          clean: true

      - name: Verify reviewed dispatch revision
        shell: bash
        run: |
          set -euo pipefail
          test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"
          test -z "$(git status --porcelain=v1)"

""",
        name="controlled-demo checkout",
    )
    text = _replace_once(
        text,
        """          python -m pip install -e . matplotlib pillow imageio-ffmpeg
""",
        """          python -m pip install . matplotlib pillow imageio-ffmpeg
""",
        name="controlled-demo non-editable install",
    )
    DEMO.write_text(text, encoding="utf-8")


def main() -> None:
    _harden_filament()
    _harden_concentration()
    _harden_demo()


if __name__ == "__main__":
    main()
