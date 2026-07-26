"""Pytest configuration for optional private integrations."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BAYESIAN_PHYSTWIN_AVAILABLE = (
    importlib.util.find_spec("bayesian_phystwin") is not None
)
_BAYESIAN_PHYSTWIN_TEST_MODULES = frozenset(
    {
        "test_causal4d_bpt_belief.py",
        "test_causal4d_molmo_acceptance.py",
        "test_causal4d_molmo_adapter.py",
        "test_causal4d_phystwin_backend.py",
        "test_causal4d_real_calibration.py",
        "test_causal4d_real_oracle_audit.py",
        "test_causal4d_rest_geometry.py",
        "test_causal4d_rest_geometry_protocol.py",
        "test_causal4d_rest_geometry_transfer.py",
        "test_discrepancy_localization_aggregate.py",
        "test_phystwin_propagated_state.py",
        "test_phystwin_resumable.py",
    }
)


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Exclude private integration modules only when their package is unavailable."""

    if _BAYESIAN_PHYSTWIN_AVAILABLE:
        return None
    return collection_path.name in _BAYESIAN_PHYSTWIN_TEST_MODULES


def pytest_report_header(config) -> str | None:
    """Make the reduced public-CI scope visible in every pytest report."""

    if _BAYESIAN_PHYSTWIN_AVAILABLE:
        return None
    return (
        "bayesian_phystwin is unavailable; excluding 12 private integration "
        "test modules"
    )
