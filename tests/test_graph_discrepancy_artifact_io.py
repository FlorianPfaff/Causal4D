from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import causal4d.discrepancy_belief as discrepancy_module
from causal4d.artifact_io import (
    load_strict_json_object,
    read_regular_file_beneath,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import (
    GraphDiscrepancyBelief,
    load_graph_discrepancy_belief,
    write_graph_discrepancy_belief,
)


def _basis() -> np.ndarray:
    return np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])


def _belief() -> GraphDiscrepancyBelief:
    return GraphDiscrepancyBelief(
        basis_sha256=array_sha256(_basis()),
        component_ids=("component-0",),
        coefficient_mean_m=np.asarray(
            [[[0.01, 0.0, 0.0], [0.0, -0.02, 0.0]]],
            dtype=np.float64,
        ),
        coefficient_covariance_m2=np.zeros((1, 3, 2, 2), dtype=np.float64),
        projection_variance_m2=np.asarray(
            [1e-6, 2e-6, 3e-6],
            dtype=np.float64,
        ),
        transition_model_id="persistence",
        innovation_model_id="unit-test",
        metadata={"future_frames_read": 0},
    )


def _write_bundle(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest = tmp_path / "belief.json"
    record = write_graph_discrepancy_belief(manifest, _belief())
    payload = tmp_path / record["payload"]["path"]
    return manifest, payload, record


def _write_manifest(manifest: Path, record: dict[str, Any]) -> None:
    manifest.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_payload_arrays(payload: Path) -> dict[str, np.ndarray]:
    with np.load(payload, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def _replace_payload(
    payload: Path,
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(payload, **arrays)
    record["payload"]["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()


def test_round_trip_preserves_identity_and_arrays(tmp_path: Path) -> None:
    manifest, _, record = _write_bundle(tmp_path)
    loaded = load_graph_discrepancy_belief(manifest)

    assert loaded.artifact_id == record["artifact_id"] == _belief().artifact_id
    np.testing.assert_array_equal(
        loaded.coefficient_mean_m,
        _belief().coefficient_mean_m,
    )
    np.testing.assert_array_equal(
        loaded.coefficient_covariance_m2,
        _belief().coefficient_covariance_m2,
    )
    np.testing.assert_array_equal(
        loaded.projection_variance_m2,
        _belief().projection_variance_m2,
    )


def test_manifest_validation_uses_the_exact_opened_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, record = _write_bundle(tmp_path)
    real_loader = discrepancy_module.load_strict_json_object

    def replacing_loader(payload: bytes, *, name: str) -> dict[str, Any]:
        manifest.write_text("{}\n", encoding="utf-8")
        return real_loader(payload, name=name)

    monkeypatch.setattr(
        discrepancy_module,
        "load_strict_json_object",
        replacing_loader,
    )
    loaded = load_graph_discrepancy_belief(manifest)

    assert loaded.artifact_id == record["artifact_id"]


def test_payload_validation_uses_the_exact_opened_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, payload, record = _write_bundle(tmp_path)
    real_loader = discrepancy_module.load_npz_bytes

    def replacing_loader(
        payload_bytes: bytes,
        *,
        name: str,
        expected_arrays: frozenset[str],
    ) -> dict[str, np.ndarray]:
        payload.write_bytes(b"concurrent replacement")
        return real_loader(
            payload_bytes,
            name=name,
            expected_arrays=expected_arrays,
        )

    monkeypatch.setattr(discrepancy_module, "load_npz_bytes", replacing_loader)
    loaded = load_graph_discrepancy_belief(manifest)

    assert loaded.artifact_id == record["artifact_id"]


def test_payload_path_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest, _, record = _write_bundle(tmp_path)
    record["payload"]["path"] = "../belief.npz"
    _write_manifest(manifest, record)

    with pytest.raises(ValueError, match="safe POSIX relative path"):
        load_graph_discrepancy_belief(manifest)


def test_payload_path_rejects_symbolic_link(tmp_path: Path) -> None:
    manifest, payload, record = _write_bundle(tmp_path)
    link = tmp_path / "linked-payload.npz"
    try:
        link.symlink_to(payload.name)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")
    record["payload"]["path"] = link.name
    _write_manifest(manifest, record)

    with pytest.raises(ValueError, match="ordinary readable file"):
        load_graph_discrepancy_belief(manifest)


def test_payload_path_rejects_intermediate_symbolic_link(tmp_path: Path) -> None:
    root = tmp_path / "root"
    real = root / "real"
    real.mkdir(parents=True)
    (real / "payload.bin").write_bytes(b"payload")
    link = root / "linked"
    try:
        link.symlink_to(real.name, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ValueError, match="ordinary readable file"):
        read_regular_file_beneath(root, "linked/payload.bin")


def test_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest, _, _ = _write_bundle(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    text = text.replace(
        '  "artifact_kind": "GraphDiscrepancyBelief",',
        (
            '  "artifact_kind": "GraphDiscrepancyBelief",\n'
            '  "artifact_kind": "GraphDiscrepancyBelief",'
        ),
        1,
    )
    manifest.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_graph_discrepancy_belief(manifest)


def test_manifest_rejects_nonfinite_json_numbers(tmp_path: Path) -> None:
    manifest, _, record = _write_bundle(tmp_path)
    record["metadata"] = {"invalid": float("nan")}
    manifest.write_text(
        json.dumps(record, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_graph_discrepancy_belief(manifest)


def test_manifest_rejects_overflowing_json_numbers() -> None:
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_strict_json_object(
            b'{"overflow": 1e400}',
            name="test manifest",
        )


def test_manifest_rejects_coercion_dependent_version(tmp_path: Path) -> None:
    manifest, _, record = _write_bundle(tmp_path)
    record["version"] = "1"
    _write_manifest(manifest, record)

    with pytest.raises(ValueError, match="version must be an integer"):
        load_graph_discrepancy_belief(manifest)


def test_manifest_rejects_unexpected_fields(tmp_path: Path) -> None:
    manifest, _, record = _write_bundle(tmp_path)
    record["unexpected"] = True
    _write_manifest(manifest, record)

    with pytest.raises(ValueError, match="fields changed"):
        load_graph_discrepancy_belief(manifest)


def test_payload_rejects_unexpected_arrays(tmp_path: Path) -> None:
    manifest, payload, record = _write_bundle(tmp_path)
    arrays = _read_payload_arrays(payload)
    arrays["unexpected"] = np.asarray([1.0], dtype=np.float64)
    _replace_payload(payload, record, arrays)
    _write_manifest(manifest, record)

    with pytest.raises(ValueError, match="array inventory changed"):
        load_graph_discrepancy_belief(manifest)


def test_payload_rejects_noncanonical_dtypes(tmp_path: Path) -> None:
    manifest, payload, record = _write_bundle(tmp_path)
    arrays = _read_payload_arrays(payload)
    arrays["coefficient_mean_m"] = arrays["coefficient_mean_m"].astype(
        np.float32
    )
    _replace_payload(payload, record, arrays)
    _write_manifest(manifest, record)

    with pytest.raises(ValueError, match="coefficient_mean_m must use float64"):
        load_graph_discrepancy_belief(manifest)


def test_manifest_itself_must_not_be_a_symbolic_link(tmp_path: Path) -> None:
    manifest, _, _ = _write_bundle(tmp_path)
    link = tmp_path / "belief-link.json"
    try:
        link.symlink_to(manifest.name)
    except (NotImplementedError, OSError):
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ValueError, match="ordinary readable file"):
        load_graph_discrepancy_belief(link)


def test_writer_rejects_npz_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not use the .npz suffix"):
        write_graph_discrepancy_belief(tmp_path / "belief.npz", _belief())
