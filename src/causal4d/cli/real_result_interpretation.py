"""Create the preregistered interpretation of real-experiment gate outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causal4d.real_result_interpretation import (
    RealResultGateSummary,
    interpret_real_result,
    write_real_result_interpretation,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args(argv)

    payload = json.loads(arguments.gate_summary.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        parser.error("gate summary JSON must contain an object")
    gates = RealResultGateSummary.from_dict(payload)
    interpretation = interpret_real_result(gates)
    write_real_result_interpretation(
        arguments.output,
        interpretation,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(interpretation.as_dict(), indent=2, sort_keys=True))
    if arguments.require_complete and interpretation.paper_status == "incomplete":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
