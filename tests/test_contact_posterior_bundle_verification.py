from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causal4d.result_bundle_verification import (
    verify_embedded_result_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(root: Path) -> Path:
    root.mkdir()
    payloads = {
        "summary.json": "{}\n",
        "contact_recovery.csv": "seed\n",
        "interventions.csv": "seed\n",
    }
    for name, content in payloads.items():
        (root / name).write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "benchmark": "causal4d-latent-contact-v1",
        "artifacts": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": _sha256(root / name),
            }
            for name in payloads
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_verified_bundle_provenance_is_portable(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    result = verify_embedded_result_bundle(bundle)

    assert result["bundle_name"] == "bundle"
    assert result["artifact_count"] == 3
    assert result["benchmark"] == "causal4d-latent-contact-v1"
    assert "directory" not in result
    assert set(result["artifacts"]) == {
        "contact_recovery.csv",
        "interventions.csv",
        "summary.json",
    }


def test_bundle_rejects_undeclared_files(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    (bundle / "undeclared.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        verify_embedded_result_bundle(bundle)


def test_bundle_rejects_symlinked_artifacts(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    payload = bundle / "summary.json"
    external = tmp_path / "external-summary.json"
    external.write_bytes(payload.read_bytes())
    payload.unlink()
    payload.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        verify_embedded_result_bundle(bundle)


def test_bundle_rejects_digest_drift(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    (bundle / "summary.json").write_text("{\"changed\": true}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="byte count|checksum"):
        verify_embedded_result_bundle(bundle)


def test_bundle_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    manifest = bundle / "manifest.json"
    manifest.write_text(
        '{"schema_version":1,"schema_version":1,'
        '"benchmark":"causal4d-latent-contact-v1","artifacts":{}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        verify_embedded_result_bundle(bundle)


def test_bundle_rejects_nonfinite_manifest_values(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    manifest = bundle / "manifest.json"
    manifest.write_text(
        '{"schema_version":1,"benchmark":"causal4d-latent-contact-v1",'
        '"artifacts":{"summary.json":{"bytes":NaN,"sha256":"'
        + "0" * 64
        + '"}}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite JSON number"):
        verify_embedded_result_bundle(bundle)
