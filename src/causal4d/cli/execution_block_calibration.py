"""Fit and evaluate registered execution-block conformal calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causal4d.execution_block_calibration import (
    ExecutionBlockCalibrationCase,
    evaluate_execution_block_cases,
    fit_execution_block_conformal_calibration,
    load_execution_block_conformal_calibration,
    save_execution_block_conformal_calibration,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported execution-block manifest schema")
    if (
        not isinstance(payload.get("outer_fold_id"), str)
        or not payload["outer_fold_id"]
    ):
        raise ValueError("execution-block manifest has no outer_fold_id")
    return payload


def _load_prediction_case(specification: Mapping[str, Any]) -> Any:
    # Reuse the existing label-loading path while keeping optional BPT imports
    # out of module import and ``--help`` execution.
    from causal4d.cli.real_calibration import _load_case

    return _load_case(dict(specification))


def _load_block_case(
    specification: Mapping[str, Any],
    *,
    outer_fold_id: str,
    split_role: str,
) -> ExecutionBlockCalibrationCase:
    case = _load_prediction_case(specification)
    execution_id = str(specification.get("execution_id", case.case_id))
    session_id = specification.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError(f"execution {execution_id!r} has no session_id")
    declared_fold = specification.get("outer_fold_id", outer_fold_id)
    if declared_fold != outer_fold_id:
        raise ValueError(f"execution {execution_id!r} belongs to another outer fold")
    return ExecutionBlockCalibrationCase.from_real_calibration_case(
        case,
        execution_id=execution_id,
        session_id=session_id,
        outer_fold_id=outer_fold_id,
        split_role=split_role,  # type: ignore[arg-type]
    )


def _artifact_metadata(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    retained = {
        key: manifest[key]
        for key in (
            "protocol_id",
            "protocol_design_sha256",
            "preacquisition_plan_id",
            "preacquisition_amendment_sha256",
            "method_freeze_sha256",
        )
        if key in manifest
    }
    retained["source_manifest_sha256"] = manifest_sha256
    return retained


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit",
        help=(
            "fit an equal-execution variance transform and a finite-rank "
            "execution-block threshold"
        ),
    )
    fit.add_argument("source_manifest_json")
    fit.add_argument("output_calibration_json")
    fit.add_argument("--confidence-level", type=float, default=0.90)
    fit.add_argument("--expected-calibration-units", type=int, default=9)
    fit.add_argument("--expected-fit-units", type=int)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a locked target fold without revising the threshold",
    )
    evaluate.add_argument("calibration_json")
    evaluate.add_argument("target_manifest_json")
    evaluate.add_argument("output_evaluation_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fit":
            manifest = _load_manifest(args.source_manifest_json)
            outer_fold_id = str(manifest["outer_fold_id"])
            fit_cases = tuple(
                _load_block_case(
                    value,
                    outer_fold_id=outer_fold_id,
                    split_role="fit",
                )
                for value in manifest.get("fit", [])
            )
            calibration_cases = tuple(
                _load_block_case(
                    value,
                    outer_fold_id=outer_fold_id,
                    split_role="calibration",
                )
                for value in manifest.get("calibration", [])
            )
            source_sha256 = _sha256(args.source_manifest_json)
            calibration = fit_execution_block_conformal_calibration(
                fit_cases,
                calibration_cases,
                confidence_level=args.confidence_level,
                expected_calibration_units=args.expected_calibration_units,
                expected_fit_units=args.expected_fit_units,
                metadata=_artifact_metadata(
                    manifest,
                    manifest_sha256=source_sha256,
                ),
            )
            save_execution_block_conformal_calibration(
                args.output_calibration_json,
                calibration,
            )
            result = {
                "passed": True,
                "calibration_id": calibration.calibration_id,
                "claim_ready": calibration.claim_ready,
                "outer_fold_id": calibration.outer_fold_id,
                "fit_execution_count": len(calibration.fit_execution_ids),
                "calibration_execution_count": len(calibration.calibration_scores),
                "order_statistic_rank_one_based": (
                    calibration.order_statistic_rank_one_based
                ),
                "threshold": calibration.threshold,
                "output": str(Path(args.output_calibration_json).resolve()),
            }
        else:
            calibration = load_execution_block_conformal_calibration(
                args.calibration_json
            )
            manifest = _load_manifest(args.target_manifest_json)
            outer_fold_id = str(manifest["outer_fold_id"])
            target_specs = manifest.get("target", manifest.get("cases", []))
            target_cases = tuple(
                _load_block_case(
                    value,
                    outer_fold_id=outer_fold_id,
                    split_role="target",
                )
                for value in target_specs
            )
            evaluation = evaluate_execution_block_cases(
                target_cases,
                calibration,
            )
            evaluation["target_manifest"] = {
                "sha256": _sha256(args.target_manifest_json),
            }
            output = Path(args.output_evaluation_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    evaluation,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            result = {
                "passed": True,
                "calibration_id": calibration.calibration_id,
                "outer_fold_id": calibration.outer_fold_id,
                "target_execution_count": evaluation["target_execution_count"],
                "execution_block_coverage": evaluation["execution_block_coverage"],
                "output": str(output.resolve()),
            }
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"passed": False, "error": str(error)},
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
