"""Import a portable external sparse trajectory forecast."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    global import_external_forecast
    global save_external_forecast

    from causal4d.external_forecast import (
        import_external_forecast,
        save_external_forecast,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and content-address an external sparse 3-D trajectory "
            "forecast for Causal4D semantic/task-posterior scoring."
        )
    )
    parser.add_argument("source_npz")
    parser.add_argument("import_manifest_json")
    parser.add_argument("output_npz")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="fail rather than replacing an existing output artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    bundle = import_external_forecast(
        args.source_npz,
        args.import_manifest_json,
    )
    save_external_forecast(
        args.output_npz,
        bundle,
        overwrite=not args.no_overwrite,
    )
    summary = bundle.summary()
    summary["output"] = str(Path(args.output_npz).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
