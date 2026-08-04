from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOAD_CONSUMERS = (
    "src/causal4d/cli/abduct_phystwin_intervention.py",
    "src/causal4d/cli/audit_parameter_support.py",
    "src/causal4d/cli/audit_real_oracle_gap.py",
    "src/causal4d/cli/evaluate_molmo_acceptance.py",
    "src/causal4d/cli/evaluate_phystwin_molmo.py",
)
SAVE_CONSUMERS = (
    "src/causal4d/cli/counterfactual_phystwin.py",
    "src/causal4d/cli/phystwin_rollout_bank.py",
)


def test_rollout_bank_commands_share_the_strict_archive_boundary() -> None:
    for relative in LOAD_CONSUMERS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "from causal4d.rollout_bank_io import load_rollout_bank" in text
        assert "from causal4d.phystwin_backend import load_rollout_bank" not in text

    for relative in SAVE_CONSUMERS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "from causal4d.rollout_bank_io import save_rollout_bank" in text
        assert "save_rollout_bank," not in text


def test_installed_wheel_golden_path_uses_strict_rollout_bank_io() -> None:
    text = (ROOT / "ci/three_repository_rollout.py").read_text(encoding="utf-8")

    assert "from causal4d.rollout_bank_io import load_rollout_bank" in text
    assert "save_rollout_bank" in text
    assert "from causal4d.phystwin_backend import (" in text
    backend_import = text.split(
        "from causal4d.phystwin_backend import (", 1
    )[1].split(")", 1)[0]
    assert "load_rollout_bank" not in backend_import
    assert "save_rollout_bank" not in backend_import
