"""CLI for equal-case discrepancy-localization aggregation."""

from __future__ import annotations

import argparse
import json


def _load_runtime_dependencies() -> None:
    """Load optional integrations only after argparse handles ``--help``."""
    global aggregate_discrepancy_localization

    from causal4d.discrepancy_localization_aggregate import (
        aggregate_discrepancy_localization,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_json")
    parser.add_argument("summary_json", nargs="+")
    args = parser.parse_args()
    _load_runtime_dependencies()
    result = aggregate_discrepancy_localization(
        args.summary_json,
        args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
