from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "run_pr193_latent_contact_stability.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_pr193_latent_contact_stability_test_module",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load latent-contact stability utility")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [True, None, float("nan"), float("inf")])
def test_registered_gate_rejects_invalid_threshold_scalars(
    tmp_path: Path,
    value: object,
) -> None:
    module = _load_script()
    path = tmp_path / "success-gates.json"
    path.write_text(
        json.dumps(
            {
                "gates": [
                    {
                        "name": "shifted_node_accuracy",
                        "comparison": ">=",
                        "threshold": value,
                    }
                ]
            },
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="threshold is invalid"):
        module._registered_gate(path, threshold=1.0)


def test_registered_gate_retains_the_frozen_numeric_threshold(
    tmp_path: Path,
) -> None:
    module = _load_script()
    path = tmp_path / "success-gates.json"
    path.write_text(
        json.dumps(
            {
                "gates": [
                    {
                        "name": "shifted_node_accuracy",
                        "comparison": ">=",
                        "threshold": 0.8,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    gate = module._registered_gate(path, threshold=0.8)

    assert gate["threshold"] == 0.8
