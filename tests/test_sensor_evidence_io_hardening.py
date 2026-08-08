from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import causal4d.sensor_evidence as sensor_evidence_module
from causal4d.artifact_io import ArtifactValidationError
from causal4d.sensor_evidence import (
    ActuatorEvidence,
    ContactWrenchEvidence,
    load_independent_sensor_evidence,
    save_independent_sensor_evidence,
)


def _identity() -> dict[str, str]:
    return {
        "protocol_id": "sensor_evidence_io_unit",
        "case_id": "unit_case",
        "observed_action_id": "u_obs",
    }


def _actuator_evidence() -> ActuatorEvidence:
    positions = np.zeros((2, 1, 3), dtype=float)
    return ActuatorEvidence(
        **_identity(),
        stream_id="measured_end_effector",
        clock_id="robot_monotonic",
        provenance="robot encoder independent of object reconstruction",
        sample_times_s=np.asarray([0.0, 1.0 / 30.0]),
        positions_m=positions,
        variance_m2=np.full_like(positions, 1.0e-4),
        evidence_frame_stop=6,
    )


def _wrench_evidence() -> ContactWrenchEvidence:
    wrench = np.asarray([[1.0, 0.0, 0.0]])
    return ContactWrenchEvidence(
        **_identity(),
        stream_id="wrist_force",
        clock_id="robot_monotonic",
        provenance="wrist force sensor independent of object reconstruction",
        sample_times_s=np.asarray([0.0]),
        wrench=wrench,
        variance=np.full_like(wrench, 1.0e-3),
        quantity_names=("force_x_n", "force_y_n", "force_z_n"),
        evidence_frame_stop=6,
    )


def _read_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _descriptor(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return json.loads(
        np.asarray(arrays["descriptor_json"], dtype=np.uint8).tobytes().decode("utf-8")
    )


def _rewrite_descriptor(
    path: Path,
    arrays: dict[str, np.ndarray],
    descriptor: dict[str, object],
) -> None:
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    arrays["descriptor_json"] = np.frombuffer(encoded, dtype=np.uint8)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def test_publication_is_exactly_once_by_default(tmp_path: Path) -> None:
    path = tmp_path / "sensor_evidence.npz"
    actuator = _actuator_evidence()
    save_independent_sensor_evidence(path, actuator)
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        save_independent_sensor_evidence(path, _wrench_evidence())

    assert path.read_bytes() == original
    restored = load_independent_sensor_evidence(path)
    assert type(restored) is ActuatorEvidence
    assert restored.artifact_id == actuator.artifact_id


def test_overwrite_requires_explicit_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "sensor_evidence.npz"
    save_independent_sensor_evidence(path, _actuator_evidence())
    wrench = _wrench_evidence()

    save_independent_sensor_evidence(path, wrench, overwrite=True)

    restored = load_independent_sensor_evidence(path)
    assert type(restored) is ContactWrenchEvidence
    assert restored.artifact_id == wrench.artifact_id


def test_unbound_extra_archive_members_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sensor_evidence.npz"
    save_independent_sensor_evidence(path, _actuator_evidence())
    arrays = _read_archive(path)
    arrays["unbound_payload"] = np.asarray([1], dtype=np.int64)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.raises(ArtifactValidationError, match="closed array inventory"):
        load_independent_sensor_evidence(path)


def test_coercible_descriptor_schema_drift_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sensor_evidence.npz"
    save_independent_sensor_evidence(path, _actuator_evidence())
    arrays = _read_archive(path)
    descriptor = _descriptor(arrays)

    descriptor["schema_version"] = 1.0
    _rewrite_descriptor(path, arrays, descriptor)
    with pytest.raises(ValueError, match="schema version"):
        load_independent_sensor_evidence(path)

    descriptor["schema_version"] = 1
    descriptor["case_id"] = 17
    _rewrite_descriptor(path, arrays, descriptor)
    with pytest.raises(ArtifactValidationError, match="case_id"):
        load_independent_sensor_evidence(path)


def test_symlinked_input_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    save_independent_sensor_evidence(source, _actuator_evidence())
    link = tmp_path / "link.npz"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ArtifactValidationError, match="ordinary readable file"):
        load_independent_sensor_evidence(link)


def test_failed_write_leaves_no_partial_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sensor_evidence.npz"

    def fail_after_partial_write(handle, **arrays) -> None:
        del arrays
        handle.write(b"partial")
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(
        sensor_evidence_module.np,
        "savez_compressed",
        fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="injected write failure"):
        save_independent_sensor_evidence(path, _actuator_evidence())

    assert not path.exists()
    assert not tuple(tmp_path.glob(".sensor_evidence.npz.*.tmp"))


def test_overwrite_flag_requires_exact_boolean(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="exact boolean"):
        save_independent_sensor_evidence(
            tmp_path / "sensor_evidence.npz",
            _actuator_evidence(),
            overwrite=1,
        )
