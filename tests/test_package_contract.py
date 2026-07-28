from pathlib import Path

import causal4d
from causal4d.provider_contract import BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE


def test_package_version_and_phystwin_range_match_pyproject() -> None:
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert f'version = "{causal4d.__version__}"' in pyproject
    assert f'"bayesian-phystwin{BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE}"' in pyproject
