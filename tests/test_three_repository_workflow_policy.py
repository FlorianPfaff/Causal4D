from __future__ import annotations

from pathlib import Path


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "bayesian-phystwin-provider-compatibility.yml"
)


def test_trusted_events_cannot_pass_by_skipping_private_repositories() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "credential-gate:" in text
    assert "Require private-repository read access" in text
    assert "exit 1" in text
    assert "needs: credential-gate" in text
    assert "steps.access.outputs.enabled" not in text
    assert "the required three-repository golden path cannot run" in text
    assert "BPT_READ_SSH_KEY" in text
    assert "ssh-key: ${{ secrets.BPT_READ_SSH_KEY }}" in text
    assert "token: ${{ secrets.BPT_READ_TOKEN }}" in text


def test_external_fork_limitation_is_separate_and_explicit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "external-pull-request:" in text
    assert "External PR cannot access private providers" in text
    assert "github.event.pull_request.head.repo.full_name != github.repository" in text
    assert "maintainers must run the installed-wheel golden path" in text


def test_strict_claim_bearing_path_is_mandatory_and_fail_closed() -> None:
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


def test_built_wheels_receive_persistent_content_identities() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Record wheel SHA-256 identities" in text
    assert "sha256sum ./*.whl | sort" in text
    assert "three-repository-wheel-sha256.txt" in text
    assert "Wheel SHA-256 identities" in text
