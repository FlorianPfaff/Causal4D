#!/usr/bin/env python3
"""Run reproducible, higher-powered Causal4D diagnostics on a research runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.cli.dynamic_contact_benchmark import delayed_contact_case
from causal4d.contact_evaluation import (
    run_latent_contact_benchmark,
    write_latent_contact_artifacts,
)
from causal4d.contact_inference import LatentContactConfig
from causal4d.evaluation import (
    run_counterfactual_benchmark,
    write_benchmark_artifacts,
)


_PROFILE_SETTINGS = {
    "smoke": {
        "primary_seeds": range(0, 5),
        "stress_seeds": range(100, 103),
        "latent_seeds": range(200, 202),
        "dynamic_seeds": range(0, 10),
        "bootstrap_resamples": 500,
    },
    "standard": {
        "primary_seeds": range(0, 50),
        "stress_seeds": range(100, 120),
        "latent_seeds": range(200, 210),
        "dynamic_seeds": range(0, 100),
        "bootstrap_resamples": 5_000,
    },
    "full": {
        "primary_seeds": range(0, 200),
        "stress_seeds": range(1_000, 1_100),
        "latent_seeds": range(2_000, 2_050),
        "dynamic_seeds": range(0, 500),
        "bootstrap_resamples": 20_000,
    },
}

_SCENARIOS = (
    (
        "nominal",
        {
            "world_control_rotation_deg": 8.0,
            "world_nonlinear_stiffening": 0.18,
            "inference_noise_std_m": 0.006,
        },
    ),
    (
        "matched_world",
        {
            "world_control_rotation_deg": 0.0,
            "world_nonlinear_stiffening": 0.0,
            "inference_noise_std_m": 0.006,
        },
    ),
    (
        "strong_world_mismatch",
        {
            "world_control_rotation_deg": 16.0,
            "world_nonlinear_stiffening": 0.36,
            "inference_noise_std_m": 0.006,
        },
    ),
    (
        "high_inference_noise",
        {
            "world_control_rotation_deg": 8.0,
            "world_nonlinear_stiffening": 0.18,
            "inference_noise_std_m": 0.012,
        },
    ),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bootstrap_mean_interval(
    values: list[float],
    *,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if array.size == 1:
        return float(array[0]), float(array[0])
    chunk_size = min(1_000, resamples)
    means: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        count = min(chunk_size, remaining)
        indices = rng.integers(0, array.size, size=(count, array.size))
        means.append(np.mean(array[indices], axis=1))
        remaining -= count
    bootstrap = np.concatenate(means)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return float(lower), float(upper)


def _seed_means(rows: list[dict[str, Any]], metric: str) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), []).append(float(row[metric]))
    return [float(np.mean(grouped[seed])) for seed in sorted(grouped)]


def _summarize_counterfactual(
    result: dict[str, Any],
    *,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    rows = list(result["interventions"])
    rng = np.random.default_rng(bootstrap_seed)
    group_summaries: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    worlds = sorted({str(row["world_condition"]) for row in rows})
    for world in worlds:
        for method in methods:
            selected = [
                row
                for row in rows
                if row["world_condition"] == world and row["method"] == method
            ]
            rmse_by_seed = _seed_means(selected, "trajectory_rmse_m")
            coverage_by_seed = _seed_means(selected, "coverage")
            nees_by_seed = _seed_means(selected, "nees")
            lower, upper = _bootstrap_mean_interval(
                rmse_by_seed,
                resamples=resamples,
                rng=rng,
            )
            group_summaries.append(
                {
                    "world_condition": world,
                    "method": method,
                    "seed_count": len(rmse_by_seed),
                    "case_count": len(selected),
                    "mean_trajectory_rmse_m": float(np.mean(rmse_by_seed)),
                    "trajectory_rmse_ci95_m": [lower, upper],
                    "mean_coverage": float(np.mean(coverage_by_seed)),
                    "mean_nees": float(np.mean(nees_by_seed)),
                    "gross_failure_fraction": float(
                        np.mean([float(row["gross_failure"]) for row in selected])
                    ),
                }
            )

    comparisons: list[dict[str, Any]] = []
    for world in worlds:
        by_key: dict[tuple[int, str, str], dict[str, float]] = {}
        for row in rows:
            if row["world_condition"] != world:
                continue
            key = (int(row["seed"]), str(row["object"]), str(row["action"]))
            by_key.setdefault(key, {})[str(row["method"])] = float(
                row["trajectory_rmse_m"]
            )
        for comparator in ("physics_only", "generative_only"):
            case_relative: list[tuple[int, float]] = []
            case_absolute: list[tuple[int, float]] = []
            for key, values in sorted(by_key.items()):
                if "hybrid" not in values or comparator not in values:
                    continue
                baseline = values[comparator]
                hybrid = values["hybrid"]
                case_absolute.append((key[0], baseline - hybrid))
                case_relative.append(
                    (key[0], (baseline - hybrid) / max(baseline, 1e-15))
                )
            seed_relative = _pair_seed_means(case_relative)
            seed_absolute = _pair_seed_means(case_absolute)
            relative_interval = _bootstrap_mean_interval(
                seed_relative,
                resamples=resamples,
                rng=rng,
            )
            absolute_interval = _bootstrap_mean_interval(
                seed_absolute,
                resamples=resamples,
                rng=rng,
            )
            comparisons.append(
                {
                    "world_condition": world,
                    "comparison": f"hybrid_vs_{comparator}",
                    "seed_count": len(seed_relative),
                    "case_count": len(case_relative),
                    "hybrid_win_fraction": float(
                        np.mean([value > 0.0 for _, value in case_absolute])
                    ),
                    "mean_relative_rmse_improvement": float(np.mean(seed_relative)),
                    "relative_improvement_ci95": list(relative_interval),
                    "mean_absolute_rmse_improvement_m": float(np.mean(seed_absolute)),
                    "absolute_improvement_ci95_m": list(absolute_interval),
                }
            )
    return {
        "groups": group_summaries,
        "paired_comparisons": comparisons,
    }


def _pair_seed_means(values: list[tuple[int, float]]) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for seed, value in values:
        grouped.setdefault(seed, []).append(value)
    return [float(np.mean(grouped[seed])) for seed in sorted(grouped)]


def _run_counterfactual_suite(
    output: Path,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_summaries: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for scenario_index, (name, overrides) in enumerate(_SCENARIOS):
        seeds = (
            list(settings["primary_seeds"])
            if name == "nominal"
            else list(settings["stress_seeds"])
        )
        config = CounterfactualBenchmarkConfig(**overrides)
        started = time.perf_counter()
        result = run_counterfactual_benchmark(seeds=seeds, config=config)
        elapsed = time.perf_counter() - started
        artifact_paths = write_benchmark_artifacts(
            result,
            output / "counterfactual" / name,
        )
        scenario_summaries.append(
            {
                "scenario": name,
                "config": config.as_dict(),
                "seeds": seeds,
                "elapsed_seconds": elapsed,
                "artifacts": artifact_paths,
                "diagnostics": _summarize_counterfactual(
                    result,
                    resamples=int(settings["bootstrap_resamples"]),
                    bootstrap_seed=20260803 + scenario_index,
                ),
            }
        )
        timings.append(
            {
                "evaluation": f"counterfactual/{name}",
                "case_count": len(result["interventions"]),
                "elapsed_seconds": elapsed,
            }
        )
    return scenario_summaries, timings


def _run_latent_contact(
    output: Path,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    seeds = list(settings["latent_seeds"])
    benchmark_config = CounterfactualBenchmarkConfig()
    contact_config = LatentContactConfig(
        parameter_particle_count=12,
        observation_fraction=0.20,
        observation_noise_std_m=0.0015,
        confidence_level=benchmark_config.confidence_level,
    )
    started = time.perf_counter()
    result = run_latent_contact_benchmark(
        seeds=seeds,
        benchmark_config=benchmark_config,
        contact_config=contact_config,
    )
    elapsed = time.perf_counter() - started
    paths = write_latent_contact_artifacts(
        result,
        output / "latent-contact",
    )
    summary = {
        "seeds": seeds,
        "elapsed_seconds": elapsed,
        "success_gates": result["success_gates"],
        "aggregate": result["aggregate"],
        "artifacts": paths,
    }
    timing = {
        "evaluation": "latent-contact",
        "case_count": _infer_case_count(result),
        "elapsed_seconds": elapsed,
    }
    return summary, timing


def _infer_case_count(result: dict[str, Any]) -> int:
    for key in ("interventions", "contact_recovery", "rows"):
        value = result.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _run_dynamic_contact(
    output: Path,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for seed in settings["dynamic_seeds"]:
        for prefix in (4, 6, 8, 10):
            rows.append(
                delayed_contact_case(
                    seed=int(seed),
                    frame_count=24,
                    prefix_frame_count=prefix,
                )
            )
    elapsed = time.perf_counter() - started
    csv_path = output / "dynamic-contact" / "cases.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"gates", "transition_config", "inference_config"}
        }
        for row in rows
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    _write_json(output / "dynamic-contact" / "cases.json", rows)
    gate_values = [bool(value) for row in rows for value in dict(row["gates"]).values()]
    summary = {
        "case_count": len(rows),
        "seed_count": len(list(settings["dynamic_seeds"])),
        "prefix_frame_counts": [4, 6, 8, 10],
        "elapsed_seconds": elapsed,
        "all_prefix_only": all(
            int(row["future_observations_read"]) == 0 for row in rows
        ),
        "all_gates_passed": all(gate_values),
        "gate_pass_fraction": float(np.mean(gate_values)),
        "mean_relative_rmse_improvement": float(
            np.mean([row["relative_rmse_improvement"] for row in rows])
        ),
        "minimum_relative_rmse_improvement": float(
            np.min([row["relative_rmse_improvement"] for row in rows])
        ),
        "mean_contact_onset_absolute_error_frames": float(
            np.mean([row["contact_onset_absolute_error_frames"] for row in rows])
        ),
        "maximum_contact_onset_absolute_error_frames": float(
            np.max([row["contact_onset_absolute_error_frames"] for row in rows])
        ),
        "mean_future_coverage": float(
            np.mean([row["future_coverage"] for row in rows])
        ),
        "artifacts": {
            "json": str(output / "dynamic-contact" / "cases.json"),
            "csv": str(csv_path),
        },
    }
    timing = {
        "evaluation": "dynamic-contact",
        "case_count": len(rows),
        "elapsed_seconds": elapsed,
    }
    return summary, timing


def _determinism_check() -> dict[str, Any]:
    config = CounterfactualBenchmarkConfig()
    first = run_counterfactual_benchmark(seeds=[91_001, 91_002], config=config)
    second = run_counterfactual_benchmark(seeds=[91_001, 91_002], config=config)
    first_digest = _canonical_digest(first)
    second_digest = _canonical_digest(second)
    return {
        "seed_count": 2,
        "first_sha256": first_digest,
        "second_sha256": second_digest,
        "passed": first_digest == second_digest,
    }


def _command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    packages = (
        "causal4d",
        "numpy",
        "scipy",
        "torch",
        "warp-lang",
        "bayesian-phystwin",
        "prob4d",
    )
    environment: dict[str, Any] = {
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {name: _package_version(name) for name in packages},
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        import torch

        environment["torch_cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except (ImportError, RuntimeError) as error:
        environment["torch_cuda"] = {"available": False, "error": str(error)}
    try:
        import warp as wp

        wp.init()
        environment["warp"] = {
            "version": getattr(wp, "__version__", None),
            "cuda_devices": [str(device) for device in wp.get_cuda_devices()],
        }
    except (ImportError, RuntimeError) as error:
        environment["warp"] = {"cuda_devices": [], "error": str(error)}
    return environment


def _write_timings(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("evaluation", "case_count", "elapsed_seconds"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Causal4D self-hosted evaluation",
        "",
        f"- Profile: `{summary['profile']}`",
        f"- Runner: `{summary['environment'].get('runner_name')}`",
        f"- Commit: `{os.environ.get('GITHUB_SHA', 'unknown')}`",
        "",
        "## Controlled counterfactual benchmark",
        "",
        (
            "| Scenario | World | Method | RMSE mm (95% bootstrap CI) "
            "| Coverage | NEES | Gross failure |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in summary["counterfactual"]:
        for row in scenario["diagnostics"]["groups"]:
            lower, upper = row["trajectory_rmse_ci95_m"]
            lines.append(
                "| {scenario} | {world} | {method} | "
                "{mean:.3f} [{lower:.3f}, {upper:.3f}] | "
                "{coverage:.3f} | {nees:.3f} | {gross:.3f} |".format(
                    scenario=scenario["scenario"],
                    world=row["world_condition"],
                    method=row["method"],
                    mean=1_000.0 * row["mean_trajectory_rmse_m"],
                    lower=1_000.0 * lower,
                    upper=1_000.0 * upper,
                    coverage=row["mean_coverage"],
                    nees=row["mean_nees"],
                    gross=row["gross_failure_fraction"],
                )
            )
    lines.extend(
        [
            "",
            "## Paired hybrid comparisons",
            "",
            (
                "| Scenario | World | Comparison | Hybrid win fraction | "
                "Mean relative RMSE improvement (95% CI) |"
            ),
            "|---|---|---|---:|---:|",
        ]
    )
    for scenario in summary["counterfactual"]:
        for row in scenario["diagnostics"]["paired_comparisons"]:
            lower, upper = row["relative_improvement_ci95"]
            lines.append(
                "| {scenario} | {world} | {comparison} | {wins:.3f} | "
                "{mean:.3%} [{lower:.3%}, {upper:.3%}] |".format(
                    scenario=scenario["scenario"],
                    world=row["world_condition"],
                    comparison=row["comparison"],
                    wins=row["hybrid_win_fraction"],
                    mean=row["mean_relative_rmse_improvement"],
                    lower=lower,
                    upper=upper,
                )
            )
    latent = summary["latent_contact"]
    dynamic = summary["dynamic_contact"]
    lines.extend(
        [
            "",
            "## Other diagnostics",
            "",
            f"- Latent-contact overall gate: "
            f"`{latent['success_gates'].get('overall_passed')}`.",
            f"- Dynamic-contact cases: `{dynamic['case_count']}`; all gates passed: "
            f"`{dynamic['all_gates_passed']}`; prefix-only: "
            f"`{dynamic['all_prefix_only']}`.",
            f"- Exact repeated-run determinism: `{summary['determinism']['passed']}`.",
            "",
            "These runs are diagnostic and do not replace the registered "
            "same-object physical experiment.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(output: Path) -> None:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        data = path.read_bytes()
        artifacts.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "artifacts": artifacts,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=sorted(_PROFILE_SETTINGS),
        default="standard",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    settings = _PROFILE_SETTINGS[arguments.profile]

    environment = _environment()
    _write_json(output / "environment.json", environment)

    counterfactual, timings = _run_counterfactual_suite(output, settings)
    latent, latent_timing = _run_latent_contact(output, settings)
    timings.append(latent_timing)
    dynamic, dynamic_timing = _run_dynamic_contact(output, settings)
    timings.append(dynamic_timing)
    determinism = _determinism_check()

    summary = {
        "schema_version": 1,
        "profile": arguments.profile,
        "environment": environment,
        "counterfactual": counterfactual,
        "latent_contact": latent,
        "dynamic_contact": dynamic,
        "determinism": determinism,
        "timings": timings,
        "claim_boundary": (
            "Diagnostic compute evaluation only; not a substitute for the "
            "registered same-object physical experiment."
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_timings(output / "timings.csv", timings)
    _write_markdown(output / "summary.md", summary)
    _write_manifest(output)

    integrity_passed = bool(determinism["passed"]) and bool(dynamic["all_prefix_only"])
    print(
        json.dumps(
            {
                "integrity_passed": integrity_passed,
                "summary": str(output / "summary.json"),
            }
        )
    )
    return 0 if integrity_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
