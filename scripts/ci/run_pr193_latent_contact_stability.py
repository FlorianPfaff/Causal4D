#!/usr/bin/env python3
"""Run a fresh, fixed-design latent-contact stability diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
from typing import Any, TextIO

import numpy as np


EXPECTED_OBJECTS = ("cloth", "rope", "soft_block")
EXPECTED_SETTING = "online_adaptation"
EXPECTED_WORLD = "shifted_contact"
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260807


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        _stream_output(process.stdout, log)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"command failed with exit code {return_code}: {' '.join(command)}"
        )


def _stream_output(stream: TextIO, log: TextIO) -> None:
    for line in stream:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()


def _parse_seed_range(value: str) -> list[int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("seed range must use START:STOP syntax")
    start, stop = (int(part) for part in parts)
    if start < 0 or stop <= start:
        raise ValueError("seed range must be nonnegative and nonempty")
    return list(range(start, stop))


def _canonical_boolean(value: str, *, name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{name} must be a canonical boolean, got {value!r}")


def _wilson_interval(successes: int, count: int) -> list[float]:
    if count <= 0 or successes < 0 or successes > count:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1.0 + z * z / count
    centre = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return [centre - radius, centre + radius]


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "cluster bootstrap requires at least two one-dimensional values"
        )
    rng = np.random.default_rng(seed)
    remaining = resamples
    means: list[np.ndarray] = []
    while remaining:
        chunk = min(1_000, remaining)
        indices = rng.integers(0, values.size, size=(chunk, values.size))
        means.append(np.mean(values[indices], axis=1))
        remaining -= chunk
    bootstrap = np.concatenate(means)
    interval = np.quantile(bootstrap, [0.025, 0.975])
    return [float(interval[0]), float(interval[1])]


def _summarize_cases(group: list[dict[str, str]]) -> dict[str, object]:
    if not group:
        raise ValueError("cannot summarize an empty case group")
    correct = np.asarray(
        [
            _canonical_boolean(row["node_correct"], name="node_correct")
            for row in group
        ],
        dtype=np.float64,
    )
    covered = np.asarray(
        [
            _canonical_boolean(
                row["node_credible_covered"],
                name="node_credible_covered",
            )
            for row in group
        ],
        dtype=np.float64,
    )
    confidence = np.asarray(
        [float(row["node_confidence"]) for row in group],
        dtype=np.float64,
    )
    truth_probability = np.asarray(
        [float(row["node_truth_probability"]) for row in group],
        dtype=np.float64,
    )
    brier = np.asarray(
        [float(row["node_brier"]) for row in group],
        dtype=np.float64,
    )
    for name, array in (
        ("node_confidence", confidence),
        ("node_truth_probability", truth_probability),
        ("node_brier", brier),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    successes = int(np.sum(correct))
    count = len(group)
    return {
        "case_count": count,
        "correct_count": successes,
        "accuracy": float(np.mean(correct)),
        "case_level_wilson_95_interval": _wilson_interval(successes, count),
        "credible_coverage": float(np.mean(covered)),
        "mean_confidence": float(np.mean(confidence)),
        "mean_truth_probability": float(np.mean(truth_probability)),
        "mean_brier": float(np.mean(brier)),
    }


def _load_shifted_online_rows(
    recovery_path: Path,
    *,
    expected_seeds: list[int],
) -> list[dict[str, str]]:
    with recovery_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required_fields = {
            "seed",
            "object",
            "setting",
            "world_condition",
            "node_correct",
            "node_credible_covered",
            "node_confidence",
            "node_truth_probability",
            "node_brier",
        }
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError("contact recovery CSV lacks required stability fields")
        rows = list(reader)
    selected = [
        row
        for row in rows
        if row["setting"] == EXPECTED_SETTING
        and row["world_condition"] == EXPECTED_WORLD
    ]
    identities = [(int(row["seed"]), row["object"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("shifted online panel contains duplicate seed/object rows")
    expected_identities = {
        (seed, object_name)
        for seed in expected_seeds
        for object_name in EXPECTED_OBJECTS
    }
    actual_identities = set(identities)
    if actual_identities != expected_identities:
        missing = sorted(expected_identities - actual_identities)[:10]
        extra = sorted(actual_identities - expected_identities)[:10]
        raise ValueError(
            f"unexpected shifted online panel: missing={missing}, extra={extra}"
        )
    return selected


def _registered_gate(success_gates_path: Path, *, threshold: float) -> dict[str, Any]:
    value = json.loads(success_gates_path.read_text(encoding="utf-8"))
    gates = value.get("gates")
    if not isinstance(gates, list):
        raise ValueError("success gate artifact lacks gates")
    matches = [gate for gate in gates if gate.get("name") == "shifted_node_accuracy"]
    if len(matches) != 1:
        raise ValueError("success gate artifact must contain one shifted-node gate")
    gate = dict(matches[0])
    if gate.get("comparison") != ">=":
        raise ValueError("shifted-node gate comparison changed")
    if not math.isclose(float(gate.get("threshold")), threshold, abs_tol=0.0):
        raise ValueError("shifted-node gate threshold changed")
    return gate


def _build_report(
    output_root: Path,
    *,
    target_sha: str,
    seed_range: str,
    expected_seeds: list[int],
    threshold: float,
) -> dict[str, Any]:
    bundle_root = output_root / "independent-seeds"
    diagnostic_path = (
        output_root / "diagnostics" / "contact-posterior-diagnostics.json"
    )
    selected = _load_shifted_online_rows(
        bundle_root / "contact_recovery.csv",
        expected_seeds=expected_seeds,
    )
    gate = _registered_gate(bundle_root / "success_gates.json", threshold=threshold)

    by_object: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_object[row["object"]].append(row)
        by_seed[int(row["seed"])].append(row)
    if any(len(by_seed[seed]) != len(EXPECTED_OBJECTS) for seed in expected_seeds):
        raise ValueError("each seed must contribute exactly one case per topology")

    overall = _summarize_cases(selected)
    if not math.isclose(
        float(overall["accuracy"]),
        float(gate["value"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("reconstructed shifted-node accuracy differs from frozen gate")

    seed_accuracies = np.asarray(
        [
            np.mean(
                [
                    _canonical_boolean(row["node_correct"], name="node_correct")
                    for row in by_seed[seed]
                ]
            )
            for seed in expected_seeds
        ],
        dtype=np.float64,
    )
    overall["independent_seed_count"] = len(expected_seeds)
    overall["seed_cluster_standard_deviation"] = float(
        np.std(seed_accuracies, ddof=1)
    )
    overall["seed_cluster_bootstrap_95_interval"] = _bootstrap_mean_interval(
        seed_accuracies,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    overall["seed_cluster_bootstrap_resamples"] = BOOTSTRAP_RESAMPLES

    block_size = 50
    if len(expected_seeds) % block_size:
        raise ValueError("seed panel must be divisible into 50-seed blocks")
    blocks: list[dict[str, object]] = []
    for block_index, offset in enumerate(range(0, len(expected_seeds), block_size)):
        block_seeds = expected_seeds[offset : offset + block_size]
        block_rows = [row for seed in block_seeds for row in by_seed[seed]]
        block = _summarize_cases(block_rows)
        block.update(
            {
                "block_index": block_index,
                "seed_start_inclusive": block_seeds[0],
                "seed_stop_exclusive": block_seeds[-1] + 1,
                "passes_frozen_threshold": float(block["accuracy"]) >= threshold,
            }
        )
        blocks.append(block)

    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    for name in ("overall", "admission_boundary", "recomputation_parity"):
        if name not in diagnostic:
            raise ValueError(f"topology diagnostic lacks {name}")
    admission = diagnostic["admission_boundary"]
    parity = diagnostic["recomputation_parity"]
    if not isinstance(admission, dict) or admission.get("passed") is not True:
        raise ValueError("contact-posterior source admission did not pass")
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        raise ValueError("contact-posterior recomputation parity did not pass")

    seed_panel_sha256 = hashlib.sha256(
        json.dumps(expected_seeds, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DLatentContactStabilityDiagnostic",
        "target_sha": target_sha,
        "seed_range": seed_range,
        "seed_count": len(expected_seeds),
        "seed_panel_sha256": seed_panel_sha256,
        "frozen_node_accuracy_threshold": threshold,
        "method_or_threshold_changed": False,
        "frozen_gate_record": gate,
        "overall_shifted_online": overall,
        "threshold_margin": float(overall["accuracy"]) - threshold,
        "passes_frozen_threshold_on_this_diagnostic_panel": (
            float(overall["accuracy"]) >= threshold
        ),
        "per_object": {
            name: _summarize_cases(by_object[name]) for name in EXPECTED_OBJECTS
        },
        "consecutive_50_seed_blocks": blocks,
        "block_pass_count": sum(
            bool(block["passes_frozen_threshold"]) for block in blocks
        ),
        "block_count": len(blocks),
        "topology_diagnostic_overall": diagnostic["overall"],
        "admission_boundary": admission,
        "recomputation_parity": parity,
        "claim_boundary": (
            "Fresh-seed diagnostic only. It does not change or rescue the "
            "registered exact-node threshold, estimator, physical protocol, "
            "or real-evidence claim boundary."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    overall = report["overall_shifted_online"]
    threshold = float(report["frozen_node_accuracy_threshold"])
    cluster_interval = overall["seed_cluster_bootstrap_95_interval"]
    lines = [
        "# Frozen latent-contact stability diagnostic",
        "",
        f"- Target: `{report['target_sha']}`",
        f"- Fresh seeds: `{report['seed_range']}` ({report['seed_count']})",
        "- Estimator/configuration/threshold changed: `false`",
        f"- Shifted exact-node accuracy: `{overall['accuracy']:.4%}` "
        f"({overall['correct_count']}/{overall['case_count']})",
        "- Seed-cluster bootstrap 95% interval: "
        f"`[{cluster_interval[0]:.4%}, {cluster_interval[1]:.4%}]`",
        f"- Frozen threshold: `{threshold:.2%}`",
        f"- Threshold margin: `{report['threshold_margin']:.4%}`",
        "- Consecutive 50-seed blocks passing threshold: "
        f"`{report['block_pass_count']}/{report['block_count']}`",
        "",
        "## Per topology",
        "",
        "| Object | Correct | Accuracy | Wilson 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for name in EXPECTED_OBJECTS:
        item = report["per_object"][name]
        interval = item["case_level_wilson_95_interval"]
        lines.append(
            f"| {name} | {item['correct_count']}/{item['case_count']} | "
            f"{item['accuracy']:.2%} | [{interval[0]:.2%}, {interval[1]:.2%}] |"
        )
    lines.extend(
        [
            "",
            "This panel characterizes fresh-seed stability only. The registered "
            "gate remains authoritative and cannot be revised by this diagnostic.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--seeds", default="3000:3500")
    parser.add_argument("--threshold", type=float, default=0.80)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_root = args.target_root.resolve()
    output_root = args.output_root.resolve()
    expected_seeds = _parse_seed_range(args.seeds)
    output_root.mkdir(parents=True, exist_ok=True)

    bundle_root = output_root / "independent-seeds"
    _run(
        [
            sys.executable,
            "-m",
            "causal4d.cli.latent_contact_benchmark",
            "--output-dir",
            str(bundle_root),
            "--seeds",
            args.seeds,
            "--frames",
            "56",
            "--training-repeats",
            "2",
            "--parameter-grid-count",
            "5",
            "--contact-parameter-particles",
            "12",
            "--observation-fraction",
            "0.20",
            "--observation-noise-mm",
            "1.5",
        ],
        cwd=target_root,
        log_path=output_root / "benchmark-console.log",
    )
    _run(
        [
            sys.executable,
            str(target_root / "scripts" / "ci" / "verify_result_bundle.py"),
            str(bundle_root / "manifest.json"),
        ],
        cwd=target_root,
        log_path=output_root / "bundle-verification.log",
    )
    diagnostic_root = output_root / "diagnostics"
    _run(
        [
            sys.executable,
            str(target_root / "scripts" / "ci" / "analyze_contact_posterior.py"),
            str(bundle_root),
            "--output-dir",
            str(diagnostic_root),
            "--diffusion-strength",
            "1.0",
            "--force-field-equivalence-threshold",
            "0.90",
        ],
        cwd=target_root,
        log_path=output_root / "diagnostic-console.log",
    )

    report = _build_report(
        output_root,
        target_sha=args.target_sha,
        seed_range=args.seeds,
        expected_seeds=expected_seeds,
        threshold=args.threshold,
    )
    stability_root = output_root / "stability"
    _write_json(stability_root / "summary.json", report)
    _write_markdown(stability_root / "summary.md", report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
