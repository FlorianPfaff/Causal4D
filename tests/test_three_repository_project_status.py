from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CI_ROOT = ROOT / "ci"
STATUS = CI_ROOT / "project_status_v1.json"
GOLDEN_PATH = CI_ROOT / "three_repository_golden_path.py"
if str(CI_ROOT) not in sys.path:
    sys.path.insert(0, str(CI_ROOT))

from three_repository_status import validate_causal4d_status  # noqa: E402


def _write_status(tmp_path: Path, mutation) -> Path:
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "project-status.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_project_status_matches_causal4d_and_provider_contract() -> None:
    summary = validate_causal4d_status(STATUS)

    assert summary["status_id"] == "causal4d-project-status-v1"
    assert summary["claim_status"] == "controlled_passed_real_pending"
    assert summary["versions"] == {"causal4d": "0.5.0"}
    assert len(summary["status_sha256"]) == 64


def test_project_status_rejects_claim_inflation(tmp_path: Path) -> None:
    path = _write_status(
        tmp_path,
        lambda payload: payload.__setitem__("claim_status", "real_confirmed"),
    )

    with pytest.raises(RuntimeError, match="overstates"):
        validate_causal4d_status(path)


def test_project_status_rejects_provider_range_drift(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        payload["packages"]["bayesian-phystwin"]["supported_versions"] = ">=0.5,<0.6"

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="compatibility range drifted"):
        validate_causal4d_status(path)


def test_project_status_rejects_causal4d_version_drift(tmp_path: Path) -> None:
    def mutate(payload) -> None:
        payload["packages"]["causal4d"]["required_version"] = "9.0.0"

    path = _write_status(tmp_path, mutate)
    with pytest.raises(RuntimeError, match="installed Causal4D version differs"):
        validate_causal4d_status(path)


def test_installed_wheel_golden_path_binds_project_status() -> None:
    text = GOLDEN_PATH.read_text(encoding="utf-8")

    assert "validate_installed_stack_status" in text
    assert '"project_status": project_status' in text
    assert "project_status_v1.json" in text
