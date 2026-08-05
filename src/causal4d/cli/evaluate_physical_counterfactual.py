"""Evaluate a PhysicalPosterior against an identity-bound physical target."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

EVALUATION_SCHEMA = "causal4d.physical_counterfactual_evaluation"
EVALUATION_SCHEMA_VERSION = 1


def _load_runtime_dependencies() -> None:
    """Load command dependencies only after argparse handles ``--help``."""

    global atomic_write_binary
    global PhysicalPosterior
    global load_contract
    global load_physical_target
    global evaluate_beta_zero_physical_posterior

    from causal4d.atomic_io import atomic_write_binary
    from causal4d.contracts import PhysicalPosterior, load_contract
    from causal4d.physical_target import load_physical_target
    from causal4d.physical_validation import evaluate_beta_zero_physical_posterior


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a discrepancy-aware physical posterior at beta=0 against "
            "a strict, non-pickled physical target artifact."
        )
    )
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("physical_target_npz")
    parser.add_argument("output_json")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing result instead of enforcing exactly-once output",
    )
    return parser


def _evaluation_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"physical evaluation output path contains a symlink: {current}"
            )


def _publish_result(
    path: str | Path,
    result: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    target = Path(path)
    _reject_symlink_components(target)
    serialized = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    def writer(handle: BinaryIO) -> None:
        handle.write(serialized)

    def validate(temporary: Path) -> None:
        try:
            restored = json.loads(
                temporary.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("physical evaluation result is not valid JSON") from error
        if not isinstance(restored, dict) or restored != dict(result):
            raise ValueError("physical evaluation result changed during serialization")
        candidate = dict(restored)
        artifact_id = candidate.pop("evaluation_id", None)
        if artifact_id != _evaluation_id(candidate):
            raise ValueError("physical evaluation result has an invalid content ID")

    atomic_write_binary(
        target,
        writer,
        overwrite=overwrite,
        validate=validate,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    if not np.isfinite(args.confidence_level) or not 0.0 < args.confidence_level < 1.0:
        raise ValueError("--confidence-level must lie strictly between zero and one")

    posterior = load_contract(args.physical_posterior_npz)
    if not isinstance(posterior, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    target = load_physical_target(args.physical_target_npz)
    truth, mask = target.aligned_for_posterior(posterior)
    evaluation_target = target.evaluation_target(start_frame=args.start_frame)

    metrics = evaluate_beta_zero_physical_posterior(
        posterior,
        truth,
        mask=mask,
        start_frame=args.start_frame,
        confidence_level=args.confidence_level,
    )
    result: dict[str, Any] = {
        **metrics,
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "case": posterior.context.case_id,
        "causal_context": posterior.context.as_dict(),
        "physical_posterior_id": posterior.artifact_id,
        "source_query_id": posterior.source_query_id,
        "physical_target_id": target.artifact_id,
        "evaluation_target_id": evaluation_target.artifact_id,
        "evaluation_target": evaluation_target.as_dict(),
        "source_final_data_sha256": target.source_final_data_sha256,
        "confidence_level": float(args.confidence_level),
    }
    result["evaluation_id"] = _evaluation_id(result)
    _publish_result(
        args.output_json,
        result,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
