from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "ci" / "compare_result_bundles.py"
WRITE_REPRODUCTION = ROOT / "scripts" / "ci" / "write_reproduction_manifest.py"
PAYLOAD_NAMES = (
    "contact_recovery.csv",
    "fold_calibration.csv",
    "interventions.csv",
    "protocol.json",
    "success_gates.json",
    "summary.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _refresh_result_manifest(directory: Path) -> None:
    artifacts = {
        name: {
            "bytes": (directory / name).stat().st_size,
            "sha256": _sha256(directory / name),
        }
        for name in PAYLOAD_NAMES
    }
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "benchmark": "causal4d-reproducibility-test-v1",
            "artifacts": artifacts,
        },
    )


def _write_bundle(directory: Path) -> None:
    directory.mkdir(parents=True)
    _write_json(
        directory / "protocol.json",
        {
            "schema_version": 1,
            "seed": 1,
            "coefficient": 0.25,
            "labels": ["rope", "cloth"],
        },
    )
    _write_json(
        directory / "success_gates.json",
        {
            "derived": {"metric": 0.5000000000005},
            "gates": [
                {
                    "name": "metric",
                    "comparison": ">=",
                    "threshold": 0.5,
                    "value": 0.5000000000005,
                    "passed": True,
                }
            ],
            "overall_passed": True,
        },
    )
    _write_json(
        directory / "summary.json",
        {
            "schema_version": 1,
            "metric": 1.0,
            "direction_error_deg": 0.0,
            "category": "controlled",
        },
    )
    _write_csv(
        directory / "contact_recovery.csv",
        [
            "seed",
            "object",
            "setting",
            "node_truth",
            "node_map",
            "node_correct",
            "node_confidence",
        ],
        [
            [1, "rope", "pre_intervention", 2, 2, True, 0.9],
            [2, "cloth", "online_adaptation", 3, 3, True, 0.8],
        ],
    )
    _write_csv(
        directory / "fold_calibration.csv",
        ["seed", "held_out_object", "contact_hypothesis_count", "prior"],
        [
            [
                1,
                "rope",
                72,
                json.dumps(
                    {
                        "source_condition_count": 4,
                        "shift_probability": 0.5,
                    },
                    sort_keys=True,
                ),
            ]
        ],
    )
    _write_csv(
        directory / "interventions.csv",
        [
            "seed",
            "method",
            "direction_error_deg",
            "trajectory_rmse_m",
            "gross_failure",
        ],
        [
            [1, "latent_contact", 0.0, 0.001, False],
            [2, "nominal_physics", 8.0, 0.004, False],
        ],
    )
    _refresh_result_manifest(directory)


def _copy_bundle(source: Path, target: Path) -> None:
    shutil.copytree(source, target)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_comparison(
    expected: Path,
    actual: Path,
    output_directory: Path,
    *extra_arguments: str,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    index = len(list(output_directory.glob("comparison-*.json")))
    output = output_directory / f"comparison-{index}.json"
    process = subprocess.run(
        [
            sys.executable,
            str(COMPARE),
            str(expected),
            str(actual),
            "--output",
            str(output),
            *extra_arguments,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert output.is_file(), process.stderr
    return process, _read_json(output)
