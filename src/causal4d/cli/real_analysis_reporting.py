"""Build registered session-clustered real-experiment effect reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causal4d.real_analysis_reporting import (
    build_real_analysis_effect_report,
    write_real_analysis_effect_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("effect_table", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-estimable", action="store_true")
    arguments = parser.parse_args(argv)

    report = build_real_analysis_effect_report(
        arguments.effect_table,
        arguments.protocol,
        method_freeze_path=arguments.method_freeze,
        analysis_manifest_path=arguments.analysis_manifest,
    )
    write_real_analysis_effect_report(
        arguments.output,
        report,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if (
        arguments.require_estimable
        and not report["primary_session_clustered_effect"]["estimable"]
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
