"""Create the preregistered interpretation of real-experiment gate outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causal4d.atomic_io import atomic_write_json
from causal4d.real_result_interpretation import (
    RealResultGateSummary,
    interpret_real_result,
    write_real_result_interpretation,
)
from causal4d.real_result_source_verification import verify_real_result_sources


def _default_verification_path(output: Path) -> Path:
    suffix = output.suffix
    stem = output.name[: -len(suffix)] if suffix else output.name
    return output.with_name(f"{stem}.sources.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--source-verification-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args(argv)

    payload = json.loads(arguments.gate_summary.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        parser.error("gate summary JSON must contain an object")
    gates = RealResultGateSummary.from_dict(payload)
    verification = verify_real_result_sources(
        gates,
        method_freeze_path=arguments.method_freeze,
        analysis_manifest_path=arguments.analysis_manifest,
    )
    verification_output = (
        arguments.source_verification_output
        or _default_verification_path(arguments.output)
    )
    if verification_output.resolve() == arguments.output.resolve():
        parser.error("source-verification output must differ from interpretation output")
    if not arguments.overwrite:
        existing = [
            str(path)
            for path in (arguments.output, verification_output)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "real-result evidence output already exists: " + ", ".join(existing)
            )

    interpretation = interpret_real_result(gates)
    atomic_write_json(
        verification_output,
        verification,
        overwrite=arguments.overwrite,
    )
    write_real_result_interpretation(
        arguments.output,
        interpretation,
        overwrite=arguments.overwrite,
    )
    console = interpretation.as_dict()
    console["source_verification"] = verification
    console["source_verification_path"] = str(verification_output)
    print(json.dumps(console, indent=2, sort_keys=True))
    if arguments.require_complete and interpretation.paper_status == "incomplete":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
