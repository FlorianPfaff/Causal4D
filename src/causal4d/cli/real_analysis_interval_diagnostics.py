"""CLI for non-decision-making real-analysis interval diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causal4d.real_analysis_interval_diagnostics import (
    build_real_analysis_interval_diagnostics,
    write_real_analysis_interval_diagnostics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build source-verified Student-t and bootstrap-t companion intervals "
            "without changing the frozen primary percentile report."
        )
    )
    parser.add_argument("effect_table", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build and publish the companion interval artifact."""

    arguments = _parser().parse_args(argv)
    payload = build_real_analysis_interval_diagnostics(
        arguments.effect_table,
        arguments.protocol,
        method_freeze_path=arguments.method_freeze,
        analysis_manifest_path=arguments.analysis_manifest,
    )
    write_real_analysis_interval_diagnostics(
        arguments.output,
        payload,
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            {
                "artifact_kind": payload["artifact_kind"],
                "diagnostic_id": payload["diagnostic_id"],
                "output": str(arguments.output),
                "primary_interval_unchanged": True,
                "sensitivity_intervals_may_change_primary_decision": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
