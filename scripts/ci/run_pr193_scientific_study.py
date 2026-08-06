#!/usr/bin/env python3
"""Run the claim-bounded scientific study for Causal4D PR 193.

The runner performs three independent tasks without changing the frozen method:

1. discover and validate any registered physical dataset available on the
   self-hosted runner, then build session-clustered reports only from complete,
   content-addressed effect tables;
2. run a source-only power and finite-calibration fragility audit for the locked
   18/12-session design; and
3. execute the existing full controlled robustness suite.

Only compact summaries are published. Raw target measurements are never copied
into the repository or Actions artifacts by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.stats import nct, t

from causal4d.real_analysis_reporting import (
    build_real_analysis_effect_report,
    write_real_analysis_effect_report,
)

PROTOCOL_ID = "causal4d-sloth-multi-action-v1"
PROTOCOL_DESIGN_SHA256 = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
TARGET_OBJECT_ID = "sloth_plush_instance_1"
TARGET_SHA = "fa6a64b2442474321e453e9e8fdccd591e0a282d"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _run(
    command: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _bounded_walk(root: Path, *, maximum_depth: int = 6):
    if not root.is_dir():
        return
    for directory, child_names, file_names in os.walk(root):
        current = Path(directory)
        try:
            depth = len(current.relative_to(root).parts)
        except ValueError:
            depth = maximum_depth + 1
        if depth >= maximum_depth:
            child_names[:] = []
        yield current, child_names, file_names


def discover_dataset(output: Path) -> tuple[Path | None, dict[str, Any]]:
    explicit = [
        Path("/data/causal4d-sloth-multi-action-v1"),
        Path("/mnt/lexar4tb/datasets/causal4d-sloth-multi-action-v1"),
        Path("/mnt/lexar4tb/datasets/Causal4D/causal4d-sloth-multi-action-v1"),
        Path("/mnt/lexar4tb/datasets/Causal4D/sloth_multi_action_v1"),
        Path("/mnt/lexar4tb/causal4d-sloth-multi-action-v1"),
    ]
    roots = [
        Path("/data"),
        Path("/mnt/lexar4tb/datasets"),
        Path("/mnt/lexar4tb"),
        Path("/srv"),
        Path("/home/github-runner"),
    ]
    candidates: set[Path] = {path for path in explicit if path.is_dir()}
    for root in roots:
        try:
            iterator = _bounded_walk(root)
            if iterator is None:
                continue
            for current, _children, file_names in iterator:
                lowered = current.name.casefold()
                if (
                    "causal4d" in lowered
                    and "sloth" in lowered
                    and ("action" in lowered or "multi" in lowered)
                ):
                    candidates.add(current)
                if "method_freeze.json" in file_names:
                    candidates.add(current)
        except OSError:
            continue

    records: list[dict[str, Any]] = []
    for candidate in sorted(candidates):
        try:
            execution_manifests = list(candidate.glob("executions/*/manifest.json"))
            session_manifests = list(candidate.glob("sessions/*/session.json"))
            json_count = sum(1 for _ in candidate.rglob("*.json"))
            record: dict[str, Any] = {
                "path": str(candidate),
                "method_freeze_present": (candidate / "method_freeze.json").is_file(),
                "freeze_validation_present": (
                    candidate / "method_freeze_validation.json"
                ).is_file(),
                "object_registration_present": (
                    candidate / "object_registration.json"
                ).is_file(),
                "slip_pilot_present": (candidate / "slip_pilot.json").is_file(),
                "timebase_calibration_present": (
                    candidate / "timebase_calibration.json"
                ).is_file(),
                "execution_manifest_count": len(execution_manifests),
                "session_manifest_count": len(session_manifests),
                "json_file_count": json_count,
            }
            record["selection_score"] = [
                int(record["method_freeze_present"]),
                min(record["execution_manifest_count"], 36),
                min(record["session_manifest_count"], 18),
                min(record["json_file_count"], 10_000),
            ]
            records.append(record)
        except OSError as error:
            records.append({"path": str(candidate), "error": str(error)})

    usable = [record for record in records if "selection_score" in record]
    selected_record = (
        max(usable, key=lambda record: tuple(record["selection_score"]))
        if usable
        else None
    )
    selected = Path(selected_record["path"]) if selected_record else None
    inventory = {
        "schema_version": 1,
        "artifact_kind": "Causal4DRunnerDatasetInventory",
        "runner_name": os.environ.get("RUNNER_NAME"),
        "candidate_count": len(records),
        "selected_dataset": str(selected) if selected else None,
        "candidates": records,
        "raw_target_data_uploaded": False,
    }
    _write_json(output / "dataset-inventory.json", inventory)
    return selected, inventory


def _git_output(root: Path, arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolve_frozen_checkout(
    dataset: Path | None,
    target_root: Path,
    work_root: Path,
) -> tuple[Path, dict[str, Any], Path | None]:
    expected_commit: str | None = None
    if dataset is not None:
        freeze = _load_json(dataset / "method_freeze.json")
        if freeze is not None:
            value = freeze.get("causal4d", {}).get("commit_sha")
            if isinstance(value, str) and len(value) == 40:
                expected_commit = value

    frozen_root = target_root
    worktree: Path | None = None
    error: str | None = None
    if expected_commit:
        try:
            exists = subprocess.run(
                ["git", "cat-file", "-e", f"{expected_commit}^{{commit}}"],
                cwd=target_root,
                check=False,
            ).returncode == 0
            if not exists:
                subprocess.run(
                    ["git", "fetch", "--no-tags", "origin", expected_commit],
                    cwd=target_root,
                    check=True,
                )
            worktree = Path(
                tempfile.mkdtemp(prefix="causal4d-pr193-frozen-", dir=work_root)
            )
            worktree.rmdir()
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), expected_commit],
                cwd=target_root,
                check=True,
            )
            frozen_root = worktree
        except (OSError, subprocess.CalledProcessError) as caught:
            error = f"{type(caught).__name__}: {caught}"
            frozen_root = target_root
            worktree = None

    actual_commit = _git_output(frozen_root, ["rev-parse", "HEAD"])
    dirty = bool(_git_output(frozen_root, ["status", "--porcelain", "--untracked-files=all"]))
    descriptor = {
        "schema_version": 1,
        "artifact_kind": "Causal4DFrozenCheckoutResolution",
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "commit_matches": expected_commit is not None and expected_commit == actual_commit,
        "clean_worktree": not dirty,
        "used_target_head_fallback": expected_commit is None or error is not None,
        "resolution_error": error,
    }
    _write_json(work_root / "frozen-checkout.json", descriptor)
    return frozen_root, descriptor, worktree


def run_real_evidence_gate(
    dataset: Path | None,
    *,
    target_root: Path,
    frozen_root: Path,
    output: Path,
) -> dict[str, Any]:
    if dataset is None:
        result = {
            "schema_version": 1,
            "artifact_kind": "Causal4DRealEvidenceRunnerGate",
            "dataset_found": False,
            "claim_ready": False,
            "analysis_ready": False,
            "blockers": [
                "registered physical dataset not found on the self-hosted runner"
            ],
        }
        _write_json(output / "real-evidence-runner-gate.json", result)
        return result

    protocol = target_root / "configs/causal4d/sloth_multi_action_v1.json"
    commands = {
        "observational": [
            "causal4d",
            "protocol",
            "real",
            "status",
            str(protocol),
            str(dataset),
            "--output-json",
            str(output / "evidence-status-observational.json"),
        ],
        "strict": [
            "causal4d",
            "protocol",
            "real",
            "status",
            str(protocol),
            str(dataset),
            "--repository-root",
            str(frozen_root),
            "--verify-file-hashes",
            "--output-json",
            str(output / "evidence-status-strict.json"),
        ],
        "readiness": [
            "causal4d",
            "protocol",
            "readiness",
            "status",
            str(frozen_root),
            str(dataset),
            "--verify-file-hashes",
            "--output-json",
            str(output / "preacquisition-readiness.json"),
        ],
        "validate_dataset": [
            "causal4d",
            "protocol",
            "real",
            "validate-dataset",
            str(protocol),
            str(dataset),
        ],
    }
    return_codes = {
        name: _run(command, log_path=output / f"{name}.log")
        for name, command in commands.items()
    }
    strict = _load_json(output / "evidence-status-strict.json")
    observational = _load_json(output / "evidence-status-observational.json")
    status = strict or observational or {}
    result = {
        "schema_version": 1,
        "artifact_kind": "Causal4DRealEvidenceRunnerGate",
        "dataset_found": True,
        "dataset": str(dataset),
        "return_codes": return_codes,
        "acquisition_complete": status.get("acquisition_complete"),
        "evidence_complete": status.get("evidence_complete"),
        "analysis_ready": bool(status.get("analysis_ready", False)),
        "full_registered_power": status.get("full_registered_power"),
        "claim_ready": bool(status.get("claim_ready", False)),
        "blockers": status.get("blockers"),
    }
    _write_json(output / "real-evidence-runner-gate.json", result)
    return result


def run_registered_reports(
    dataset: Path | None,
    *,
    target_root: Path,
    output: Path,
) -> dict[str, Any]:
    reports_root = output / "registered-reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DRegisteredArtifactIndex",
        "dataset_found": dataset is not None,
        "artifact_kind_counts": {},
        "effect_tables": [],
        "registered_analysis_manifests": [],
        "report_attempts": [],
    }
    if dataset is None:
        _write_json(output / "registered-artifact-index.json", result)
        return result

    kinds: Counter[str] = Counter()
    effect_tables: list[Path] = []
    analysis_by_sha: dict[str, Path] = {}
    freezes_by_sha: dict[str, Path] = {}
    json_files: list[Path] = []
    try:
        for position, path in enumerate(dataset.rglob("*.json")):
            if position >= 20_000:
                break
            json_files.append(path)
    except OSError:
        pass

    for path in json_files:
        try:
            if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
                continue
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                continue
            kind = payload.get("artifact_kind")
            if isinstance(kind, str):
                kinds[kind] += 1
            digest = hashlib.sha256(raw).hexdigest()
            if kind == "Causal4DRealAnalysisEffectTable":
                effect_tables.append(path)
            elif kind == "Causal4DRegisteredRealAnalysisManifest":
                analysis_by_sha[digest] = path
            elif (
                payload.get("milestone_id") == "causal4d-same-object-real-v1"
                and payload.get("status") == "sealed"
            ):
                freezes_by_sha[digest] = path
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue

    result["artifact_kind_counts"] = dict(sorted(kinds.items()))
    result["effect_tables"] = [str(path) for path in sorted(effect_tables)]
    result["registered_analysis_manifests"] = [
        str(path) for path in sorted(analysis_by_sha.values())
    ]
    protocol = target_root / "configs/causal4d/sloth_multi_action_v1.json"
    for position, table_path in enumerate(sorted(effect_tables)):
        attempt: dict[str, Any] = {"effect_table": str(table_path), "success": False}
        try:
            payload = json.loads(table_path.read_text(encoding="utf-8"))
            analysis_path = analysis_by_sha.get(payload["analysis_manifest_sha256"])
            freeze_path = freezes_by_sha.get(payload["method_freeze_sha256"])
            if analysis_path is None:
                raise ValueError("matching registered analysis manifest is absent")
            if freeze_path is None:
                raise ValueError("matching sealed method freeze is absent")
            report = build_real_analysis_effect_report(
                table_path,
                protocol,
                method_freeze_path=freeze_path,
                analysis_manifest_path=analysis_path,
            )
            report_path = reports_root / (
                f"{position:02d}-{payload.get('endpoint', 'endpoint')}-"
                f"{payload.get('metric_id', 'metric')}.json"
            )
            write_real_analysis_effect_report(report_path, report)
            attempt.update(
                {
                    "success": True,
                    "report": str(report_path),
                    "report_id": report["report_id"],
                    "endpoint": payload.get("endpoint"),
                    "metric_id": payload.get("metric_id"),
                    "estimable": report["primary_session_clustered_effect"][
                        "estimable"
                    ],
                }
            )
        except Exception as error:  # report every fail-closed reason
            attempt["error"] = f"{type(error).__name__}: {error}"
        result["report_attempts"].append(attempt)

    _write_json(output / "registered-artifact-index.json", result)
    return result


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


def run_power_and_fragility_audit(output: Path) -> dict[str, Any]:
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

    factual_sensitivity = []
    for correlation in (0.0, 0.25, 0.5, 0.75, 0.9):
        averaging_sd = math.sqrt((1.0 + correlation) / 2.0)
        for execution_effect in effect_grid:
            session_effect = execution_effect / averaging_sd
            factual_sensitivity.append(
                {
                    "within_session_execution_correlation": correlation,
                    "standardized_execution_effect": execution_effect,
                    "induced_standardized_session_mean_effect": session_effect,
                    "approximate_power": _power(18, session_effect),
                }
            )

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
    distributions = {
        "half_normal": np.abs(rng.normal(size=(draws, 9))),
        "lognormal_sigma_0.25": rng.lognormal(0.0, 0.25, size=(draws, 9)),
        "lognormal_sigma_0.50": rng.lognormal(0.0, 0.50, size=(draws, 9)),
        "lognormal_sigma_1.00": rng.lognormal(0.0, 1.00, size=(draws, 9)),
        "absolute_t5": np.abs(rng.standard_t(df=5, size=(draws, 9))),
    }
    calibration_rows = []
    for name, values in distributions.items():
        ordered = np.sort(values, axis=1)
        maximum = ordered[:, -1]
        second = ordered[:, -2]
        sample_median = np.median(values, axis=1)
        population_median = max(float(np.median(values)), 1e-15)
        normalized_threshold = maximum / population_median
        maximum_to_median = maximum / np.maximum(sample_median, 1e-15)
        second_to_maximum = second / np.maximum(maximum, 1e-15)
        calibration_rows.append(
            {
                "score_distribution": name,
                "simulation_draws": draws,
                "calibration_units": 9,
                "registered_rank_one_based": 9,
                "threshold_is_sample_maximum": True,
                "normalized_threshold_quantiles": dict(
                    zip(
                        ("q05", "q50", "q90", "q95", "q99"),
                        map(
                            float,
                            np.quantile(
                                normalized_threshold,
                                [0.05, 0.50, 0.90, 0.95, 0.99],
                            ),
                        ),
                    )
                ),
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
                        map(float, np.quantile(second_to_maximum, [0.05, 0.50, 0.95])),
                    )
                ),
                "leave_one_session_out_nominal_90_percent_threshold_finite": False,
                "fragility_may_select_or_change_threshold": False,
            }
        )

    result = {
        "schema_version": 1,
        "artifact_kind": "Causal4DRegisteredDesignPowerAndFragilityAudit",
        "protocol_id": PROTOCOL_ID,
        "protocol_design_sha256": PROTOCOL_DESIGN_SHA256,
        "session_cluster_is_independent_unit": True,
        "endpoint_power": endpoint_power,
        "factual_two_execution_session_sensitivity": factual_sensitivity,
        "precision_and_minimum_detectable_effect": precision,
        "execution_block_calibration_fragility": calibration_rows,
        "interpretation": {
            "power_values_are_standardized_design_scenarios": True,
            "not_conditioned_on_target_outcomes": True,
            "does_not_change_registered_method_or_threshold": True,
            "does_not_substitute_for_the_36_execution_physical_result": True,
        },
    }
    _write_json(output / "registered-design-power-fragility.json", result)
    lines = [
        "# Registered-design power and calibration-fragility audit",
        "",
        "This source-only diagnostic does not change the frozen estimator, threshold, exclusions, or analysis.",
        "",
        "| Sessions | 95% half-width (session SD) | d for 80% power | d for 90% power |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in precision:
        lines.append(
            "| {independent_sessions} | "
            "{expected_95_percent_half_width_in_session_sd_units:.3f} | "
            "{minimum_standardized_effect_for_80_percent_power:.3f} | "
            "{minimum_standardized_effect_for_90_percent_power:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The registered 90% execution-block threshold with nine calibration sessions is rank 9 of 9, the sample maximum. With one session removed, eight units cannot yield a finite formal 90% threshold without the registered infinite sentinel.",
            "",
        ]
    )
    (output / "registered-design-power-fragility.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result


def run_controlled_study(target_root: Path, output: Path) -> dict[str, Any]:
    qualification_rc = _run(
        [
            sys.executable,
            str(target_root / "scripts/ci/workstation2_gpu_qualification.py"),
            "--output",
            str(output / "gpu-qualification.json"),
        ],
        log_path=output / "gpu-qualification.log",
    )
    controlled_root = output / "controlled-full"
    controlled_rc = _run(
        [
            sys.executable,
            str(target_root / "scripts/ci/run_self_hosted_evaluation.py"),
            "--profile",
            "full",
            "--output-dir",
            str(controlled_root),
        ],
        log_path=output / "controlled-full.log",
    )
    return {
        "gpu_qualification_returncode": qualification_rc,
        "controlled_full_returncode": controlled_rc,
        "gpu_qualification": _load_json(output / "gpu-qualification.json"),
        "controlled_summary": _load_json(controlled_root / "summary.json"),
    }


def publish_compact_result(
    *,
    work_root: Path,
    publish_root: Path,
    summary: dict[str, Any],
) -> None:
    if publish_root.exists():
        shutil.rmtree(publish_root)
    publish_root.mkdir(parents=True, exist_ok=True)
    _write_json(publish_root / "summary.json", summary)
    candidates = [
        "dataset-inventory.json",
        "frozen-checkout.json",
        "real-evidence-runner-gate.json",
        "evidence-status-observational.json",
        "evidence-status-strict.json",
        "preacquisition-readiness.json",
        "registered-artifact-index.json",
        "registered-design-power-fragility.json",
        "registered-design-power-fragility.md",
        "gpu-qualification.json",
        "controlled-full/summary.json",
        "controlled-full/summary.md",
        "controlled-full/manifest.json",
        "controlled-full/timings.csv",
    ]
    for relative in candidates:
        source = work_root / relative
        if source.is_file() and source.stat().st_size <= 8 * 1024 * 1024:
            destination = publish_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    reports = work_root / "registered-reports"
    if reports.is_dir():
        shutil.copytree(reports, publish_root / "registered-reports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--publish-root", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    target_root = arguments.target_root.resolve()
    work_root = arguments.work_root.resolve()
    publish_root = arguments.publish_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    actual_sha = _git_output(target_root, ["rev-parse", "HEAD"])
    if actual_sha != TARGET_SHA:
        raise SystemExit(f"target SHA changed: expected {TARGET_SHA}, got {actual_sha}")
    if _git_output(target_root, ["status", "--porcelain", "--untracked-files=all"]):
        raise SystemExit("target checkout is not clean")

    dataset, inventory = discover_dataset(work_root)
    frozen_root, frozen_descriptor, worktree = resolve_frozen_checkout(
        dataset, target_root, work_root
    )
    try:
        gate = run_real_evidence_gate(
            dataset,
            target_root=target_root,
            frozen_root=frozen_root,
            output=work_root,
        )
        reports = run_registered_reports(
            dataset, target_root=target_root, output=work_root
        )
        power_audit = run_power_and_fragility_audit(work_root)
        controlled = run_controlled_study(target_root, work_root)
    finally:
        if worktree is not None:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=target_root,
                check=False,
            )

    successes = [
        item
        for item in reports.get("report_attempts", [])
        if isinstance(item, dict) and item.get("success") is True
    ]
    failures = [
        item
        for item in reports.get("report_attempts", [])
        if isinstance(item, dict) and item.get("success") is not True
    ]
    controlled_summary = controlled.get("controlled_summary") or {}
    qualification = controlled.get("gpu_qualification") or {}
    if gate.get("claim_ready"):
        scientific_priority = (
            "registered physical analysis executed or an exact effect-table blocker was recorded"
        )
    elif gate.get("dataset_found"):
        scientific_priority = (
            "complete the recorded readiness/evidence blockers before confirmatory interpretation"
        )
    else:
        scientific_priority = (
            "the registered physical dataset is not present on workstation2; acquisition remains the claim-changing milestone"
        )

    summary = {
        "schema_version": 1,
        "artifact_kind": "Causal4DPR193ScientificExecutionSummary",
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "target_sha": actual_sha,
        "dataset_found": gate.get("dataset_found"),
        "selected_dataset": inventory.get("selected_dataset"),
        "frozen_checkout": frozen_descriptor,
        "acquisition_complete": gate.get("acquisition_complete"),
        "evidence_complete": gate.get("evidence_complete"),
        "analysis_ready": gate.get("analysis_ready"),
        "full_registered_power": gate.get("full_registered_power"),
        "claim_ready": gate.get("claim_ready"),
        "registered_effect_table_count": len(reports.get("effect_tables", [])),
        "successful_registered_report_count": len(successes),
        "failed_registered_report_count": len(failures),
        "registered_report_ids": [item.get("report_id") for item in successes],
        "gpu_qualification_returncode": controlled[
            "gpu_qualification_returncode"
        ],
        "gpu_qualification_passed": qualification.get("passed"),
        "controlled_full_returncode": controlled["controlled_full_returncode"],
        "controlled_full_completed": bool(controlled_summary),
        "controlled_full_integrity": controlled_summary.get("integrity"),
        "design_power_audit_completed": bool(power_audit),
        "precision_and_minimum_detectable_effect": power_audit[
            "precision_and_minimum_detectable_effect"
        ],
        "scientific_priority": scientific_priority,
        "claim_boundary": (
            "Controlled and design diagnostics are not substitutes for the registered 18-session/36-execution physical result. A positive physical report remains bounded to sloth_plush_instance_1 and the frozen action/contact/hardware configuration."
        ),
    }
    publish_compact_result(
        work_root=work_root,
        publish_root=publish_root,
        summary=summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
