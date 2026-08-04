from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import causal4d.contact_posterior_provenance as provenance


_MANIFEST = "a" * 64
_PROTOCOL = [
    {
        "object": "cloth",
        "source_objects": "rope",
        "graph_node_count": 4,
        "sensor_node_count": 2,
        "contact_state_count": 8,
        "test_action_id": "single_lift",
    },
    {
        "object": "rope",
        "source_objects": "cloth",
        "graph_node_count": 6,
        "sensor_node_count": 2,
        "contact_state_count": 8,
        "test_action_id": "single_lift",
    },
]


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


def _source_report(manifest_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "causal4d-latent-contact-v1",
        "manifest_sha256": manifest_sha256,
        "artifact_count": 6,
        "seed_count": 1,
        "contact_recovery_row_count": 2,
        "intervention_row_count": 4,
        "online_case_count": 2,
        "unit_interval_tolerance": 1e-12,
        "passed": True,
    }


def _bundle(root: Path) -> Path:
    root.mkdir()
    _write_json(
        root / "summary.json",
        {
            "schema_version": 1,
            "benchmark": "causal4d-latent-contact-v1",
            "protocol": _PROTOCOL,
        },
    )
    _write_json(root / "protocol.json", _PROTOCOL)
    _write_csv(
        root / "contact_recovery.csv",
        ("object", "source_objects", "held_out_topology"),
        [
            ("cloth", "rope", True),
            ("rope", "cloth", True),
        ],
    )
    _write_csv(
        root / "interventions.csv",
        ("object", "source_objects", "held_out_topology"),
        [
            ("cloth", "rope", True),
            ("rope", "cloth", True),
        ],
    )
    _write_csv(
        root / "fold_calibration.csv",
        ("object", "source_objects"),
        [
            ("cloth", "rope"),
            ("rope", "cloth"),
        ],
    )
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    return root


def _install_source_verifier(
    monkeypatch: pytest.MonkeyPatch,
    bundle: Path,
    *,
    manifest_sha256: str | None = None,
) -> None:
    actual_manifest = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    report = _source_report(manifest_sha256 or actual_manifest)
    monkeypatch.setattr(
        provenance,
        "verify_contact_posterior_source_bundle",
        lambda _: report,
    )


def _rewrite_csv_value(
    bundle: Path,
    *,
    filename: str,
    field: str,
    value: str,
) -> None:
    path = bundle / filename
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    rows[1][rows[0].index(field)] = value
    _write_csv(path, tuple(rows[0]), [tuple(row) for row in rows[1:]])


def test_valid_provenance_contract_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _install_source_verifier(monkeypatch, bundle)

    report = provenance.verify_contact_posterior_provenance_bundle(bundle)

    contract = report["provenance_contract"]
    assert isinstance(contract, dict)
    assert contract["passed"] is True
    assert contract["protocol_object_count"] == 2
    assert contract["contact_recovery_row_count"] == 2
    assert contract["intervention_row_count"] == 2
    assert contract["fold_calibration_row_count"] == 2
    assert contract["nonfinite_json_rejected"] is True
    assert contract["protocol_payload_bound"] is True
    assert contract["source_topology_identities_bound"] is True
    assert contract["held_out_declarations_bound"] is True


def test_nonfinite_summary_json_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    summary = bundle / "summary.json"
    summary.write_text(
        summary.read_text(encoding="utf-8").rstrip()[:-1]
        + ', "unused": NaN}\n',
        encoding="utf-8",
    )
    _install_source_verifier(monkeypatch, bundle)

    with pytest.raises(ValueError, match="non-finite JSON number"):
        provenance.verify_contact_posterior_provenance_bundle(bundle)


def test_protocol_payload_disagreement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    changed = [dict(row) for row in _PROTOCOL]
    changed[0]["test_action_id"] = "different_action"
    _write_json(bundle / "protocol.json", changed)
    _install_source_verifier(monkeypatch, bundle)

    with pytest.raises(ValueError, match="summary and protocol.json disagree"):
        provenance.verify_contact_posterior_provenance_bundle(bundle)


def test_false_held_out_declaration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _rewrite_csv_value(
        bundle,
        filename="contact_recovery.csv",
        field="held_out_topology",
        value="False",
    )
    _install_source_verifier(monkeypatch, bundle)

    with pytest.raises(ValueError, match="held_out_topology must be True"):
        provenance.verify_contact_posterior_provenance_bundle(bundle)


def test_source_topology_identity_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _rewrite_csv_value(
        bundle,
        filename="interventions.csv",
        field="source_objects",
        value="rope;extra",
    )
    _install_source_verifier(monkeypatch, bundle)

    with pytest.raises(ValueError, match="source_objects differs from the summary"):
        provenance.verify_contact_posterior_provenance_bundle(bundle)


def test_manifest_identity_disagreement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    _install_source_verifier(monkeypatch, bundle, manifest_sha256=_MANIFEST)

    with pytest.raises(ValueError, match="disagree on manifest identity"):
        provenance.verify_contact_posterior_provenance_bundle(bundle)
