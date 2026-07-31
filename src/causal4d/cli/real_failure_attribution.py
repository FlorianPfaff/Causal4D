"""Aggregate real oracle audits into an execution-accounted failure report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn


def _load_runtime_dependencies() -> None:
    """Load numerical aggregation only after argparse handles ``--help``."""
    global aggregate_real_failure_attribution

    from causal4d.real_failure_attribution import aggregate_real_failure_attribution


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _load_json_object(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON input: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return payload


def _protocol_contract(path: str | Path) -> tuple[str, list[str]]:
    payload = _load_json_object(path)
    protocol_id = payload.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("protocol JSON must contain a nonempty protocol_id")
    executions = payload.get("executions")
    if not isinstance(executions, list) or not executions:
        raise ValueError("protocol JSON must contain a nonempty executions array")
    case_ids = []
    for index, execution in enumerate(executions):
        if not isinstance(execution, dict):
            raise ValueError(f"protocol execution {index} must be an object")
        case_id = execution.get("execution_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"protocol execution {index} lacks execution_id")
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("protocol execution IDs must be unique")
    return protocol_id, case_ids


def _exclusions(path: str | Path, *, protocol_id: str | None) -> dict[str, str]:
    payload = _load_json_object(path)
    recorded_protocol = payload.get("protocol_id")
    if protocol_id is not None and recorded_protocol != protocol_id:
        raise ValueError("exclusions JSON does not match the protocol_id")
    entries = payload.get("exclusions")
    if not isinstance(entries, list):
        raise ValueError("exclusions JSON must contain an exclusions array")
    result: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"exclusion {index} must be an object")
        case = entry.get("case_id")
        reason = entry.get("reason")
        if not isinstance(case, str) or not case:
            raise ValueError(f"exclusion {index} lacks case_id")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"exclusion {case} lacks a reason")
        if case in result:
            raise ValueError(f"duplicate exclusion for {case}")
        result[case] = reason
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_json")
    parser.add_argument("output_csv")
    parser.add_argument("audit_json", nargs="+")
    parser.add_argument(
        "--protocol-json",
        help=(
            "Registered protocol whose execution IDs must be exactly accounted "
            "by audits or explicit exclusions."
        ),
    )
    parser.add_argument(
        "--exclusions-json",
        help=(
            "JSON object with protocol_id and exclusions entries containing "
            "case_id and reason."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    protocol_id = None
    expected_cases = None
    if args.protocol_json is not None:
        protocol_id, expected_cases = _protocol_contract(args.protocol_json)
    exclusions = (
        _exclusions(args.exclusions_json, protocol_id=protocol_id)
        if args.exclusions_json is not None
        else None
    )
    if exclusions and expected_cases is None:
        raise ValueError("--exclusions-json requires --protocol-json")
    result = aggregate_real_failure_attribution(
        args.audit_json,
        args.output_json,
        output_csv=args.output_csv,
        expected_case_ids=expected_cases,
        expected_protocol_id=protocol_id,
        excluded_cases=exclusions,
    )
    print(
        json.dumps(
            {
                "output_json": str(Path(args.output_json).resolve()),
                "output_csv": str(Path(args.output_csv).resolve()),
                "evidence_fingerprint": result["evidence_fingerprint"],
                "case_accounting": result["case_accounting"],
                "track_error_comparison": result["paired_comparisons"]["track_error_m"][
                    "causal4d_vs_bayesian_phystwin_nominal_z"
                ],
                "track_error_dominant_gap": result["diagnostic_gap_attribution"][
                    "track_error_m"
                ]["dominant_gap"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
