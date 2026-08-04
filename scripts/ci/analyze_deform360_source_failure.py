#!/usr/bin/env python3
"""Attribute the frozen Deform360 source-backend failure without target access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_source_failure_attribution import (
    analyze_source_failure_milestone,
    validate_source_failure_attribution,
    write_source_failure_attribution,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/causal4d_public/deform360_replication_v1.json"),
        help="Locked Deform360 replication protocol JSON.",
    )
    parser.add_argument(
        "--milestone-root",
        type=Path,
        default=Path("milestones/deform360-replication-source-backend-v1"),
        help="Frozen source-backend milestone directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the checksummed attribution JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = analyze_source_failure_milestone(
        args.protocol,
        args.milestone_root,
    )
    output = write_source_failure_attribution(args.output, result)
    reopened = json.loads(output.read_text(encoding="utf-8"))
    validation = validate_source_failure_attribution(reopened)
    summary = {
        "output": str(output.resolve()),
        "result_sha256": validation["result_sha256"],
        "object_count": validation["object_count"],
        "classification_counts": validation["classification_counts"],
        "cohort_summary": reopened["cohort_summary"],
        "target_prefix_access_permitted": reopened["decision"][
            "target_prefix_access_permitted"
        ],
        "target_future_access_permitted": reopened["decision"][
            "target_future_access_permitted"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
