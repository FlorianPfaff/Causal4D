"""Compare result bundles bytewise and under a strict semantic contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from result_bundle_compare_runner import compare_result_bundles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a regenerated result bundle with an archived bundle while "
            "keeping byte identity separate from strict numerical reproduction."
        )
    )
    parser.add_argument("expected_dir", type=Path)
    parser.add_argument("actual_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=2e-12)
    parser.add_argument("--absolute-tolerance", type=float, default=2e-15)
    parser.add_argument(
        "--direction-angle-tolerance-deg",
        type=float,
        default=2e-6,
        help=(
            "Absolute tolerance only for direction_error_deg near zero, where "
            "arccos amplifies machine-level cosine differences."
        ),
    )
    parser.add_argument("--expected-reproduction-manifest", type=Path)
    parser.add_argument("--actual-reproduction-manifest", type=Path)
    parser.add_argument(
        "--require-actual-reproduction-manifest",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = compare_result_bundles(
        arguments.expected_dir,
        arguments.actual_dir,
        relative_tolerance=arguments.relative_tolerance,
        absolute_tolerance=arguments.absolute_tolerance,
        direction_angle_tolerance_deg=arguments.direction_angle_tolerance_deg,
        expected_reproduction_manifest=arguments.expected_reproduction_manifest,
        actual_reproduction_manifest=arguments.actual_reproduction_manifest,
        require_actual_reproduction_manifest=(
            arguments.require_actual_reproduction_manifest
        ),
    )
    output = arguments.output
    if output.is_symlink():
        raise ValueError(f"comparison output must not be a symlink: {output}")
    output = output.absolute()
    for bundle_directory in (arguments.expected_dir, arguments.actual_dir):
        absolute_bundle = bundle_directory.absolute()
        if output.is_relative_to(absolute_bundle):
            raise ValueError(
                "comparison output must be outside both verified bundle "
                f"directories: {output}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["semantic_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
