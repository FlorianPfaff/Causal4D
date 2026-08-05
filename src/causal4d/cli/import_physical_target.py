"""Convert a trusted legacy PhysTwin target into a safe Causal4D artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _load_runtime_dependencies() -> None:
    """Load optional integrations only after argparse handles ``--help``."""

    global target_validity
    global PhysicalPosterior
    global load_contract
    global build_physical_target
    global save_physical_target
    global load_trusted_pickle

    from bayesian_phystwin.causal4d_provider_v1 import target_validity
    from causal4d.contracts import PhysicalPosterior, load_contract
    from causal4d.physical_target import (
        build_physical_target,
        save_physical_target,
    )
    from causal4d.trusted_pickle import load_trusted_pickle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one explicitly trusted, digest-verified legacy final_data "
            "pickle into a strict non-pickled physical evaluation target."
        )
    )
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("legacy_final_data_pickle")
    parser.add_argument("output_target_npz")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        required=True,
        help=(
            "required explicit consent: Python pickle loading can execute code; "
            "use only with a trusted producer and independently obtained digest"
        ),
    )
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="independently obtained lowercase SHA-256 of the legacy pickle",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing target instead of enforcing exactly-once output",
    )
    return parser


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("legacy final_data payload must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError("legacy final_data mapping keys must be strings")
    return value


def _required_array(data: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in data:
        raise ValueError(f"legacy final_data payload is missing {name!r}")
    return np.asarray(data[name])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()

    posterior = load_contract(args.physical_posterior_npz)
    if not isinstance(posterior, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    data = _require_mapping(
        load_trusted_pickle(
            args.legacy_final_data_pickle,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
            expected_sha256=args.expected_sha256,
        )
    )

    object_points = _required_array(data, "object_points")
    visible = _required_array(data, "object_visibilities")
    motion_valid = _required_array(data, "object_motions_valid")
    if object_points.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("legacy object_points must use float32 or float64")
    if object_points.ndim != 3 or object_points.shape[2] != 3:
        raise ValueError("legacy object_points must have shape (T, N, 3)")
    if visible.dtype != np.dtype(bool) or visible.shape != object_points.shape[:2]:
        raise ValueError("legacy object_visibilities must have boolean shape (T, N)")
    if motion_valid.dtype != np.dtype(bool):
        raise ValueError("legacy object_motions_valid must use the boolean dtype")

    canonical_points = np.asarray(object_points, dtype=np.float32)
    valid = target_validity(visible, motion_valid)
    if np.asarray(valid).dtype != np.dtype(bool):
        raise ValueError("target_validity must return a boolean array")
    bundle = build_physical_target(
        posterior.context,
        canonical_points,
        valid,
        source_final_data_sha256=args.expected_sha256,
        metadata={
            "importer": "causal4d evidence physical-target import-legacy",
            "source_keys": [
                "object_points",
                "object_visibilities",
                "object_motions_valid",
            ],
        },
    )
    bundle.aligned_for_posterior(posterior)
    save_physical_target(
        args.output_target_npz,
        bundle,
        overwrite=args.overwrite,
    )
    result = {
        **bundle.summary(),
        "output": str(Path(args.output_target_npz).resolve()),
        "unsafe_pickle_consent": True,
        "source_digest_verified_before_unpickling": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
