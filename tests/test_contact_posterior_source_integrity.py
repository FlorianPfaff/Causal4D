from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from causal4d.contact_posterior_source_integrity import (
    verify_contact_posterior_source_bundle,
)


_ARTIFACT_NAMES = (
    "summary.json",
    "protocol.json",
    "interventions.csv",
    "contact_recovery.csv",
    "fold_calibration.csv",
    "success_gates.json",
)
_OBJECTS = ("cloth", "rope", "soft_block")
_NODES = {"cloth": "0", "rope": "5", "soft_block": "4"}
_RECOVERY_HEADER = (
    "seed",
    "object",
    "source_objects",
    "world_condition",
    "setting",
    "observation_fraction",
    "node_truth",
    "node_map",
    "node_correct",
    "node_confidence",
    "node_truth_probability",
    "node_brier",
    "node_credible_covered",
    "delay_map",
    "delay_map_correct",
    "joint_effective_sample_size",
    "joint_normalized_entropy",
)
_INTERVENTION_HEADER = (
    "seed",
    "object",
    "source_objects",
    "world_condition",
    "setting",
    "method",
    "observation_fraction",
    "trajectory_rmse_m",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: list[tuple[Any, ...]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_manifest(bundle: Path) -> None:
    artifacts: dict[str, dict[str, str | int]] = {}
    for name in _ARTIFACT_NAMES:
        payload = (bundle / name).read_bytes()
        artifacts[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    _write_json(
        bundle / "manifest.json",
        {
            "schema_version": 1,
            "benchmark": "causal4d-latent-contact-v1",
            "artifacts": artifacts,
        },
    )


def _valid_bundle(bundle: Path) -> Path:
    success_gates = {"overall_passed": False, "gates": []}
    _write_json(
        bundle / "summary.json",
        {
            "schema_version": 1,
            "benchmark": "causal4d-latent-contact-v1",
            "seeds": [100],
            "benchmark_config": {},
            "contact_config": {"observation_fraction": 0.2},
            "aggregate": {},
            "success_gates": success_gates,
        },
    )
    _write_json(bundle / "protocol.json", {})
    _write_json(bundle / "success_gates.json", success_gates)
    _write_csv(bundle / "fold_calibration.csv", ("seed",), [(100,)])

    recovery_rows: list[tuple[Any, ...]] = []
    intervention_rows: list[tuple[Any, ...]] = []
    for object_name in _OBJECTS:
        source_objects = ";".join(
            candidate for candidate in _OBJECTS if candidate != object_name
        )
        node = _NODES[object_name]
        recovery_rows.append(
            (
                100,
                object_name,
                source_objects,
                "shifted_contact",
                "online_adaptation",
                0.2,
                node,
                node,
                True,
                0.9,
                0.8,
                0.1,
                True,
                2,
                True,
                5.0,
                0.4,
            )
        )
        for method, rmse in (
            ("nominal_physics", 0.004),
            ("latent_contact", 0.001),
        ):
            intervention_rows.append(
                (
                    100,
                    object_name,
                    source_objects,
                    "shifted_contact",
                    "online_adaptation",
                    method,
                    0.2,
                    rmse,
                )
            )
    _write_csv(bundle / "contact_recovery.csv", _RECOVERY_HEADER, recovery_rows)
    _write_csv(bundle / "interventions.csv", _INTERVENTION_HEADER, intervention_rows)
    _write_manifest(bundle)
    return bundle


def _rewrite_recovery_value(
    bundle: Path,
    *,
    field: str,
    value: str,
) -> None:
    path = bundle / "contact_recovery.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    field_index = rows[0].index(field)
    rows[1][field_index] = value
    _write_csv(
        path,
        tuple(rows[0]),
        [tuple(row) for row in rows[1:]],
    )
    _write_manifest(bundle)


def test_valid_source_bundle_passes(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)

    report = verify_contact_posterior_source_bundle(bundle)

    assert report["passed"] is True
    assert report["benchmark"] == "causal4d-latent-contact-v1"
    assert report["artifact_count"] == 6
    assert report["seed_count"] == 1
    assert report["online_case_count"] == 3
    assert report["unit_interval_tolerance"] == 1e-12


def test_machine_scale_probability_overshoot_is_admitted(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    _rewrite_recovery_value(
        bundle,
        field="node_confidence",
        value="1.0000000000000002",
    )

    report = verify_contact_posterior_source_bundle(bundle)

    assert report["passed"] is True


def test_substantive_probability_overshoot_is_rejected(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    _rewrite_recovery_value(
        bundle,
        field="node_confidence",
        value="1.0000000001",
    )

    with pytest.raises(ValueError, match=r"node_confidence must be in \[0, 1\]"):
        verify_contact_posterior_source_bundle(bundle)


def test_payload_tampering_is_rejected_before_parsing(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "contact_recovery.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte count changed|checksum changed"):
        verify_contact_posterior_source_bundle(bundle)


def test_rehashed_duplicate_case_is_rejected(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "contact_recovery.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    _write_manifest(bundle)

    with pytest.raises(ValueError, match="duplicate contact-recovery row key"):
        verify_contact_posterior_source_bundle(bundle)


def test_rehashed_unpaired_trajectory_case_is_rejected(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "interventions.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    retained = [
        line for line in lines if not ("cloth" in line and "latent_contact" in line)
    ]
    path.write_text("\n".join(retained) + "\n", encoding="utf-8")
    _write_manifest(bundle)

    with pytest.raises(ValueError, match="missing paired methods"):
        verify_contact_posterior_source_bundle(bundle)


def test_rehashed_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    summary = bundle / "summary.json"
    summary.write_text(
        "{"
        '"schema_version":1,'
        '"benchmark":"causal4d-latent-contact-v1",'
        '"seeds":[100],'
        '"seeds":[101],'
        '"benchmark_config":{},'
        '"contact_config":{"observation_fraction":0.2},'
        '"aggregate":{},'
        '"success_gates":{"overall_passed":false,"gates":[]}'
        "}\n",
        encoding="utf-8",
    )
    _write_manifest(bundle)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_contact_posterior_source_bundle(bundle)
