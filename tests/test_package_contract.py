from importlib.metadata import version
from pathlib import Path

import causal4d
from causal4d.provider_contract import BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE


def test_package_version_and_phystwin_range_match_pyproject() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = { attr = "causal4d.__version__" }' in pyproject
    assert '\nversion = "0.5.0"\n' not in pyproject
    assert version("causal4d") == causal4d.__version__
    assert f'"bayesian-phystwin{BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE}"' in pyproject
