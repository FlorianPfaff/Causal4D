import json
from pathlib import Path

import pytest

from causal4d.cli.execution_block_calibration import _artifact_metadata, _load_manifest


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "outer_fold_id": "outer-fold-1",
        "protocol_id": "causal4d-sloth-multi-action-v1",
        "protocol_design_sha256": "1" * 64,
        "preacquisition_plan_id": "causal4d-sloth-preacquisition-v4",
        "preacquisition_amendment_sha256": "2" * 64,
    }


def test_manifest_loader_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        _load_manifest(path)


def test_source_metadata_requires_frozen_protocol_binding() -> None:
    manifest = _manifest()
    del manifest["preacquisition_plan_id"]
    with pytest.raises(ValueError, match="preacquisition_plan_id"):
        _artifact_metadata(manifest, manifest_sha256="3" * 64)


def test_source_metadata_binds_exact_manifest_bytes() -> None:
    metadata = _artifact_metadata(_manifest(), manifest_sha256="3" * 64)
    assert metadata["source_manifest_sha256"] == "3" * 64
    assert metadata["protocol_design_sha256"] == "1" * 64
