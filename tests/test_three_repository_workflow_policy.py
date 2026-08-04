from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / "bayesian-phystwin-provider-compatibility.yml"
)
COMMON = ROOT / "ci" / "three_repository_common.py"
MANIFEST = ROOT / "ci" / "three_repository_manifest.py"
GOLDEN_PATH = ROOT / "ci" / "three_repository_golden_path.py"


def test_trusted_events_cannot_pass_by_skipping_private_repositories() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "credential-gate:" in text
    assert "Require private-repository read access" in text
    assert "exit 1" in text
    assert "needs: credential-gate" in text
    assert "steps.access.outputs.enabled" not in text
    assert "the required three-repository golden path cannot run" in text


def test_external_fork_limitation_is_separate_and_explicit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "external-pull-request:" in text
    assert "External PR cannot access private providers" in text
    assert "github.event.pull_request.head.repo.full_name != github.repository" in text
    assert "maintainers must run the installed-wheel golden path" in text


def test_transferred_repository_locations_are_used_consistently() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    golden_path = GOLDEN_PATH.read_text(encoding="utf-8")

    current_repositories = (
        "IPS-Stuttgart/Causal4D",
        "IPS-Stuttgart/BayesianPhysTwin",
        "IPS-Stuttgart/Prob4D",
    )
    for repository in current_repositories:
        assert repository in common

    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in workflow
    assert "repository: IPS-Stuttgart/Prob4D" in workflow
    assert "FlorianPfaff/Bayesian-PhysTwin" not in workflow
    assert "FlorianPfaff/Prob4D" not in workflow
    assert "FlorianPfaff/Causal4D" not in manifest
    assert "FlorianPfaff/" not in golden_path


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
