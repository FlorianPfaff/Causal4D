#!/usr/bin/env python3
"""Run and publish the source-only registered-design audit for PR 193."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from run_pr193_hosted_push_study import build_design_audit


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    audit = build_design_audit()
    _write_json(output / "registered-design-power-fragility.json", audit)
    summary = {
        "schema_version": 1,
        "artifact_kind": "Causal4DPR193RegisteredDesignAuditSummary",
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "protocol_id": audit["protocol_id"],
        "protocol_design_sha256": audit["protocol_design_sha256"],
        "simulation_draws_per_score_family": audit[
            "execution_block_calibration_fragility"
        ][0]["simulation_draws"],
        "score_family_count": len(
            audit["execution_block_calibration_fragility"]
        ),
        "precision_and_minimum_detectable_effect": audit[
            "precision_and_minimum_detectable_effect"
        ],
        "claim_boundary": (
            "Source-only preregistered-design diagnostic. It does not use target "
            "outcomes, alter the registered threshold, or substitute for the "
            "18-session/36-execution physical result."
        ),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
