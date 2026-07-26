"""Aggregate guarded action-propagated state development diagnostics."""

from __future__ import annotations

import argparse
import json


def _load_runtime_dependencies() -> None:
    """Load optional integrations only after argparse handles ``--help``."""
    global aggregate_guarded_propagated_state_cases

    from causal4d.phystwin_propagated_state import (
        aggregate_guarded_propagated_state_cases,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_json")
    parser.add_argument("summary_json", nargs="+")
    args = parser.parse_args()
    _load_runtime_dependencies()
    result = aggregate_guarded_propagated_state_cases(
        args.summary_json,
        args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
