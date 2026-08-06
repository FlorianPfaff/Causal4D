#!/usr/bin/env python3
"""Run the source-only design audit and controlled PR 193 replication.

This execution helper is intentionally outside the immutable target under test.
It checks the target SHA before running, never reads physical target data, and
cannot modify the registered estimator, protocol, split, threshold, or gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import numpy as np
from scipy.stats import nct, t

TARGET_SHA = "fa6a64b2442474321e453e9e8fdccd591e0a282d"
PROTOCOL_ID = "causal4d-sloth-multi-action-v1"
PROTOCOL_SHA256 = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _power(n: int, standardized_effect: float, *, alpha: float = 0.05) -> float:
    degrees = n - 1
    critical = float(t.ppf(1.0 - alpha / 2.0, degrees))
    noncentrality = standardized_effect * math.sqrt(n)
    return float(
        nct.cdf(-critical, degrees, noncentrality)
        + 1.0
        - nct.cdf(critical, degrees, noncentrality)
    )


def _minimum_effect(n: int, target_power: float) -> float:
    lower, upper = 0.0, 4.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if _power(n, middle) >= target_power:
            upper = middle
        else:
            lower = middle
    return upper


def _calibration_family(
    name: str,
    factory: Callable[[], np.ndarray],
    *,
    draws: int,
) -> dict[str, Any]:
    values = factory()
    ordered = np.sort(values, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2]
    sample_median = np.median(values, axis=1)
    maximum_to_median = maximum / np.maximum(sample_median, 1e-15)
    second_to_maximum = second / np.maximum(maximum, 1e-15)
    return {
        "score_distribution": name,
        "simulation_draws": draws,
        "calibration_units": 9,
        "registered_rank_one_based": 9,
        "threshold_is_sample_maximum": True,
        "max_to_sample_median_quantiles": dict(
            zip(
                ("q50", "q90", "q95", "q99"),
                map(
                    float,
                    np.quantile(
                        maximum_to_median,
                        [0.50, 0.90, 0.95, 0.99],
                    ),
                ),
            )
        ),
        "probability_max_exceeds_twice_sample_median": float(
            np.mean(maximum_to_median > 2.0)
        ),
        "probability_max_exceeds_three_times_sample_median": float(
            np.mean(maximum_to_median > 3.0)
        ),
        "second_largest_to_maximum_quantiles": dict(
            zip(
                ("q05", "q50", "q95"),
                map(
                    float,
                    np.quantile(second_to_maximum, [0.05, 0.50, 0.95]),
                ),
            )
        ),
        "leave_one_session_out_nominal_90_percent_threshold_finite": False,
        "fragility_may_select_or_change_threshold": False,
    }


def build_design_audit() -> dict[str, Any]:
    effect_grid = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
    endpoint_power = [
        {
            "endpoint": endpoint,
            "independent_sessions": sessions,
            "standardized_session_effect": effect,
            "two_sided_alpha": 0.05,
            "approximate_power": _power(sessions, effect),
        }
        for endpoint, sessions in (
            ("factual_continuation", 18),
            ("same_grasp_transfer", 18),
            ("new_contact_transfer", 12),
        )
        for effect in effect_grid
    ]
    precision = [
        {
            "independent_sessions": sessions,
            "expected_95_percent_half_width_in_session_sd_units": float(
                t.ppf(0.975, sessions - 1) / math.sqrt(sessions)
            ),
            "minimum_standardized_effect_for_80_percent_power": _minimum_effect(
                sessions, 0.80
            ),
            "minimum_standardized_effect_for_90_percent_power": _minimum_effect(
                sessions, 0.90
            ),
        }
        for sessions in (12, 18)
    ]

    rng = np.random.default_rng(20_260_806)
    draws = 200_000
    factories: tuple[tuple[str, Callable[[], np.ndarray]], ...] = (
        ("half_normal", lambda: np.abs(rng.normal(size=(draws, 9)))),
        (
            "lognormal_sigma_0.25",
            lambda: rng.lognormal(0.0, 0.25, size=(draws, 9)),
        ),
        (
            "lognormal_sigma_0.50",
            lambda: rng.lognormal(0.0, 0.50, size=(draws, 9)),
        ),
        (
            "lognormal_sigma_1.00",
            lambda: rng.lognormal(0.0, 1.00, size=(draws, 9)),
        ),
        (
            "absolute_t5",
            lambda: np.abs(rng.standard_t(df=5, size=(draws, 9))),
        ),
    )
    calibration = [
        _calibration_family(name, factory, draws=draws)
        for name, factory in factories
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DRegisteredDesignPowerAndFragilityAudit",
        "protocol_id": PROTOCOL_ID,
        "protocol_design_sha256": PROTOCOL_SHA256,
        "session_cluster_is_independent_unit": True,
        "endpoint_power": endpoint_power,
        "precision_and_minimum_detectable_effect": precision,
        "execution_block_calibration_fragility": calibration,
        "interpretation": {
            "not_conditioned_on_target_outcomes": True,
            "does_not_change_registered_method_or_threshold": True,
            "does_not_substitute_for_the_36_execution_physical_result": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard", "full"),
        default="standard",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    target = arguments.target_root.resolve()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    actual_sha = _git_output(target, "rev-parse", "HEAD")
    if actual_sha != TARGET_SHA:
        raise SystemExit(f"target SHA changed: expected {TARGET_SHA}, got {actual_sha}")
    if _git_output(target, "status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("immutable target checkout is dirty")

    audit = build_design_audit()
    _write_json(output / "registered-design-power-fragility.json", audit)

    controlled_root = output / f"controlled-{arguments.profile}"
    console = output / f"controlled-{arguments.profile}.log"
    with console.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [
                sys.executable,
                str(target / "scripts/ci/run_self_hosted_evaluation.py"),
                "--profile",
                arguments.profile,
                "--output-dir",
                str(controlled_root),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    controlled_summary_path = controlled_root / "summary.json"
    controlled_summary = (
        json.loads(controlled_summary_path.read_text(encoding="utf-8"))
        if controlled_summary_path.is_file()
        else None
    )
    summary = {
        "schema_version": 1,
        "artifact_kind": "Causal4DPR193HostedScientificSummary",
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "target_sha": actual_sha,
        "profile": arguments.profile,
        "controlled_returncode": completed.returncode,
        "controlled_completed": controlled_summary is not None,
        "controlled_integrity": (
            controlled_summary.get("integrity")
            if isinstance(controlled_summary, dict)
            else None
        ),
        "precision_and_minimum_detectable_effect": audit[
            "precision_and_minimum_detectable_effect"
        ],
        "claim_boundary": (
            "Independent controlled and source-only design evidence; not a "
            "substitute for the registered 18-session/36-execution physical result."
        ),
    }
    _write_json(output / "hosted-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
