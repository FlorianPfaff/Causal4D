#!/usr/bin/env python3
"""Recompute and diagnose a controlled Causal4D contact-posterior bundle."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from causal4d.contact_posterior_admission import (
    analyze_admitted_contact_posterior_bundle,
)
from causal4d.contact_posterior_diagnostics import (
    DiagnosticConfig,
    write_contact_posterior_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-strength", type=float, default=1.0)
    parser.add_argument(
        "--force-field-equivalence-threshold",
        type=float,
        default=0.90,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DiagnosticConfig(
        diffusion_strength=args.diffusion_strength,
        force_field_equivalence_threshold=(args.force_field_equivalence_threshold),
    )
    result = analyze_admitted_contact_posterior_bundle(
        args.bundle_dir,
        config=config,
    )
    source_bundle = result.get("source_bundle")
    if not isinstance(source_bundle, dict):
        raise ValueError("diagnostic result is missing source-bundle provenance")
    source_integrity = source_bundle.get("integrity_verification")
    if not isinstance(source_integrity, dict):
        raise ValueError("diagnostic result is missing admission verification")

    paths = write_contact_posterior_diagnostics(result, args.output_dir)
    print(
        json.dumps(
            {
                "source_bundle_integrity": source_integrity,
                "admission_boundary": result["admission_boundary"],
                "recomputation_parity": result["recomputation_parity"],
                "overall": result["overall"],
                "by_topology": result["by_topology"],
                "artifacts": paths,
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
