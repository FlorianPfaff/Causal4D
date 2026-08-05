#!/usr/bin/env python3
"""Make required public-provider CI secret-free and mandatory."""

from __future__ import annotations

from pathlib import Path
import re


MAIN_CI = Path(".github/workflows/ci.yml")
THREE_REPO_CI = Path(
    ".github/workflows/bayesian-phystwin-provider-compatibility.yml"
)
RELEASE_POLICY = Path("tests/test_release_workflow_policy.py")
THREE_REPO_POLICY = Path("tests/test_three_repository_workflow_policy.py")
CI_DOCS = Path("docs/ci.md")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def _sub_once(text: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(
        pattern,
        lambda _match: replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return updated


def _update_main_ci() -> None:
    text = MAIN_CI.read_text(encoding="utf-8")
    pinned_job = '''  pinned-bpt:
    name: Pinned Bayesian-PhysTwin installed-wheel integration
    runs-on: ubuntu-latest
    steps:
      - name: Check out Causal4D
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
      - name: Set up Python
        uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: "3.12"
          cache: pip
      - name: Read exact Bayesian-PhysTwin pin
        id: pin
        run: echo "sha=$(python scripts/ci/read_bpt_pin.py)" >> "$GITHUB_OUTPUT"
      - name: Check out pinned Bayesian-PhysTwin
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          repository: IPS-Stuttgart/BayesianPhysTwin
          ref: ${{ steps.pin.outputs.sha }}
          path: _bpt
          persist-credentials: false
      - name: Build Causal4D and pinned provider wheels
        run: |
          python -m pip install --upgrade build pip
          wheelhouse="$RUNNER_TEMP/pinned-bpt-wheelhouse"
          mkdir -p "$wheelhouse"
          python -m build --wheel --outdir "$wheelhouse" .
          python -m build --wheel --outdir "$wheelhouse" ./_bpt
          test "$(find "$wheelhouse" -maxdepth 1 -name '*.whl' | wc -l)" -eq 2
      - name: Install isolated wheels
        run: |
          wheelhouse="$RUNNER_TEMP/pinned-bpt-wheelhouse"
          python -m venv artifact-venv
          artifact-venv/bin/python -m pip install --upgrade pip
          artifact-venv/bin/python -m pip install "$wheelhouse"/*.whl pytest
          artifact-venv/bin/python -m pip check
      - name: Run pinned-provider integration tests
        env:
          CAUSAL4D_REQUIRE_BPT_PROVIDER: "1"
        run: artifact-venv/bin/python scripts/ci/run_bpt_integration_tests.py
'''
    text = _sub_once(
        text,
        r"  pinned-bpt:\n.*?\n  build:\n",
        pinned_job + "\n  build:\n",
        label="pinned Bayesian-PhysTwin job",
    )
    for forbidden in ("BPT_READ_SSH_KEY", "ssh-key:"):
        if forbidden in text:
            raise RuntimeError(f"main CI still contains {forbidden}")
    MAIN_CI.write_text(text, encoding="utf-8")


def _update_three_repository_ci() -> None:
    text = THREE_REPO_CI.read_text(encoding="utf-8")
    text = _sub_once(
        text,
        r"jobs:\n  credential-gate:.*?\n  installed-wheel-golden-path:\n",
        "jobs:\n  installed-wheel-golden-path:\n",
        label="private-provider gate jobs",
    )
    text = _replace_once(
        text,
        "    needs: credential-gate\n"
        "    if: needs.credential-gate.result == 'success'\n",
        "",
        label="credential-gated job dependency",
    )
    text = _replace_once(
        text,
        "    env:\n"
        "      PROB4D_READ_TOKEN: ${{ secrets.PROB4D_READ_TOKEN }}\n",
        "",
        label="Prob4D credential environment",
    )
    text = _sub_once(
        text,
        r"    steps:\n      - name: Probe Prob4D repository access.*?"
        r"\n      - name: Check out Causal4D",
        "    steps:\n      - name: Check out Causal4D",
        label="Prob4D credential probe and skip paths",
    )
    text = text.replace(
        "        if: steps.prob4d-access.outputs.available == 'true'\n",
        "",
    )
    text = text.replace(
        "          ssh-key: ${{ secrets.BPT_READ_SSH_KEY }}\n",
        "",
    )
    text = text.replace(
        "          token: ${{ env.PROB4D_READ_TOKEN }}\n",
        "",
    )
    forbidden = (
        "credential-gate:",
        "external-pull-request:",
        "BPT_READ_SSH_KEY",
        "PROB4D_READ_TOKEN",
        "steps.prob4d-access",
        "ssh-key:",
        "needs: credential-gate",
    )
    remaining = [value for value in forbidden if value in text]
    if remaining:
        raise RuntimeError(f"three-repository CI still contains {remaining}")
    THREE_REPO_CI.write_text(text, encoding="utf-8")


def _write_policy_tests() -> None:
    RELEASE_POLICY.write_text(
        '''from pathlib import Path

import pytest


def test_tag_release_requires_public_provider_integration() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    if not workflow.is_file():
        pytest.skip("GitHub workflow is not included in the source distribution")
    text = workflow.read_text(encoding="utf-8")
    assert "Pinned Bayesian-PhysTwin installed-wheel integration" in text
    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "python -m build --wheel" in text
    assert "artifact-venv/bin/python scripts/ci/run_bpt_integration_tests.py" in text
    assert (
        "needs: [quality, core, pinned-bpt, build, installed-artifact, bundles, frozen-manifest]"
        in text
    )
    assert "BPT_READ_SSH_KEY" not in text
    assert "ssh-key:" not in text
''',
        encoding="utf-8",
    )
    THREE_REPO_POLICY.write_text(
        '''from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "bayesian-phystwin-provider-compatibility.yml"
)


def test_public_provider_workflow_is_mandatory_and_secret_free() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Prob4D -> BPT -> Causal4D installed wheels" in text
    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "repository: IPS-Stuttgart/Prob4D" in text
    assert "credential-gate:" not in text
    assert "external-pull-request:" not in text
    assert "BPT_READ_SSH_KEY" not in text
    assert "PROB4D_READ_TOKEN" not in text
    assert "ssh-key:" not in text
    assert "needs: credential-gate" not in text
    assert "steps.prob4d-access" not in text


def test_external_forks_use_the_same_public_installed_wheel_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "github.event.pull_request.head.repo.full_name" not in text
    assert "External PR cannot access private providers" not in text
    assert "private golden path" not in text


def test_strict_claim_bearing_path_is_mandatory() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Run strict claim-bearing provider-v2 admission path" in text
    assert "three_repository_provider_v2_attestation.py" in text
    assert "three-repository-provider-v2-summary.json" in text
    assert "prob4d/tests/test_claim_bearing_observation.py" in text
    assert "bayesian-phystwin/tests/test_prob4d_causal_lineage.py" in text
    assert "causal4d/tests/test_prob4d_stream_contract_version.py" in text
    assert text.count("set -o pipefail") >= 2
    assert text.count('python" -m json.tool') >= 2
    assert text.count('test -s "$RUNNER_TEMP/three-repository-') >= 2
    assert "steps.prob4d-access.outputs.available" not in text


def test_built_wheels_receive_persistent_content_identities() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Record wheel SHA-256 identities" in text
    assert "sha256sum ./*.whl | sort" in text
    assert "three-repository-wheel-sha256.txt" in text
    assert "Wheel SHA-256 identities" in text


def test_rollout_bank_contract_changes_trigger_installed_wheel_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for path in (
        "src/causal4d/rollout_bank.py",
        "src/causal4d/rollout_bank_io.py",
        "tests/test_causal4d_rollout_bank.py",
        "tests/test_rollout_bank_io.py",
    ):
        assert text.count(f'"{path}"') == 2
''',
        encoding="utf-8",
    )


def _update_docs() -> None:
    text = CI_DOCS.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "Causal4D separates its lightweight contracts from private-provider and\n"
        "hardware integrations. This keeps a base installation testable while still\n"
        "detecting drift at the Prob4D and Bayesian-PhysTwin boundaries.\n",
        "Causal4D separates its lightweight contracts from public cross-repository and\n"
        "hardware integrations. This keeps a base installation testable while still\n"
        "detecting drift at the Prob4D and Bayesian-PhysTwin boundaries.\n",
        label="CI documentation introduction",
    )
    text = _sub_once(
        text,
        r"## Private repository access\n.*?\n## Three-repository installed-wheel golden path\n",
        '''## Public repository integration

Prob4D, BayesianPhysTwin, and Causal4D are public repositories. Required
provider checks therefore use ordinary read-only `actions/checkout` steps and do
not depend on repository secrets or deploy keys. Forked pull requests exercise
the same installed-wheel boundary as organization branches, and release tags
remain blocked unless the pinned public-provider job passes. Optional hardware
workflows retain their separately documented runtime requirements.

## Three-repository installed-wheel golden path
''',
        label="private repository access documentation",
    )
    text = _sub_once(
        text,
        r"`\.github/workflows/bayesian-phystwin-provider-compatibility\.yml` is the terminal\n"
        r".*?\n\nThe installed-wheel job records",
        '''`.github/workflows/bayesian-phystwin-provider-compatibility.yml` is the terminal
Prob4D -> BayesianPhysTwin -> Causal4D compatibility check. It runs on relevant
pull requests and pushes, can be dispatched with explicit BPT and Prob4D
revisions, and runs weekly against both public repositories' `main` branches.
All events execute the full path; an unavailable checkout or contract failure is
a failing check rather than a credential-dependent skip.

The installed-wheel job records''',
        label="three-repository gate documentation",
    )
    text = _replace_once(
        text,
        "The workflow is the canonical executable specification because it can read the\n"
        "two private repositories. A local equivalent requires checkouts of all three\n"
        "repositories and a token is not needed once they are available:\n",
        "The workflow is the canonical executable specification. A local equivalent\n"
        "requires checkouts of all three public repositories:\n",
        label="local reproduction documentation",
    )
    CI_DOCS.write_text(text, encoding="utf-8")


def main() -> None:
    _update_main_ci()
    _update_three_repository_ci()
    _write_policy_tests()
    _update_docs()


if __name__ == "__main__":
    main()
