#!/usr/bin/env python3
"""Run the repository tests that exercise the Bayesian-PhysTwin boundary."""

from __future__ import annotations

import sys

import pytest


BPT_INTEGRATION_TESTS = (
    "tests/test_causal4d_bpt_belief.py",
    "tests/test_causal4d_molmo_acceptance.py",
    "tests/test_causal4d_molmo_adapter.py",
    "tests/test_causal4d_phystwin_backend.py",
    "tests/test_causal4d_real_calibration.py",
    "tests/test_causal4d_real_oracle_audit.py",
    "tests/test_causal4d_rest_geometry.py",
    "tests/test_causal4d_rest_geometry_protocol.py",
    "tests/test_causal4d_rest_geometry_transfer.py",
    "tests/test_discrepancy_localization_aggregate.py",
    "tests/test_phystwin_propagated_state.py",
)


def main() -> int:
    return pytest.main(["-ra", *BPT_INTEGRATION_TESTS])


if __name__ == "__main__":
    sys.exit(main())
