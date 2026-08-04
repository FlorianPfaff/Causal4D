from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import causal4d.rollout_bank_io as rollout_bank_io
from causal4d.rollout_bank import JointRolloutBank
from causal4d.rollout_bank_io import load_rollout_bank, save_rollout_bank


def _bank() -> JointRolloutBank:
    trajectories = np.zeros((2, 1, 4, 2, 3), dtype=np.float32)
    trajectories[1, 0, :, :, 0] = np.arange(4, dtype=np.float32)[:, None]
    return JointRolloutBank(
        hypothesis_ids=("left", "right"),
        hypothesis_metadata=(
            {
                "hypothesis_id": "left",
                "action": {"proposal_id": "left"},
            },
            {
                "hypothesis_id": "right",
                "action": {"proposal_id": "right"},
            },
        ),
        hypothesis_prior_weights=np.asarray([0.6, 0.4]),
        parameter_particles=np.asarray([[0.0, 0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=1e-6,
        confidence_level=0.9,
    )


def _legacy_archive(path: Path, bank: JointRolloutBank, manifest: dict) -> None:
    import json

    np.savez_compressed(
        path,
        hypothesis_ids=np.asarray(bank.hypothesis_ids),
        hypothesis_metadata_json=np.asarray(
            [json.dumps(value, sort_keys=True) for value in bank.hypothesis_metadata]
        ),
        hypothesis_prior_weights=bank.hypothesis_prior_weights,
        parameter_particles=bank.parameter_particles,
        parameter_weights=bank.parameter_weights,
        trajectories=bank.trajectories,
        variance_floor_m2=np.asarray(bank.variance_floor_m2),
        confidence_level=np.asarray(bank.confidence_level),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )


def _rewrite(path: Path, mutation) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    mutation(payload)
    np.savez_compressed(path, **payload)


def test_rollout_bank_round_trip_binds_content_id_and_manifest(tmp_path: Path) -> None:
    bank = _bank()
    manifest = {"source": {"revision": "a" * 40}, "count": 2}
    path = tmp_path / "bank.npz"

    save_rollout_bank(path, bank, manifest)
    restored, restored_manifest = load_rollout_bank(path)

    assert restored.artifact_id == bank.artifact_id
    assert restored_manifest == manifest
    assert np.array_equal(restored.trajectories, bank.trajectories)
    assert not restored.trajectories.flags.writeable
    with np.load(path, allow_pickle=False) as archive:
        assert str(archive["rollout_bank_id"]) == bank.artifact_id
        assert int(archive["archive_schema_version"]) == 2


def test_rollout_bank_save_refuses_overwrite(tmp_path: Path) -> None:
    first = _bank()
    changed = first.trajectories.copy()
    changed[0, 0, 0, 0, 0] = 1.0
    second = JointRolloutBank(
        hypothesis_ids=first.hypothesis_ids,
        hypothesis_metadata=first.hypothesis_metadata,
        hypothesis_prior_weights=first.hypothesis_prior_weights,
        parameter_particles=first.parameter_particles,
        parameter_weights=first.parameter_weights,
        trajectories=changed,
    )
    path = tmp_path / "bank.npz"

    save_rollout_bank(path, first, {"revision": 1}, overwrite=False)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_rollout_bank(path, second, {"revision": 2}, overwrite=False)

    assert path.read_bytes() == original
    restored, manifest = load_rollout_bank(path)
    assert restored.artifact_id == first.artifact_id
    assert manifest == {"revision": 1}


def test_rollout_bank_loader_rejects_id_and_inventory_tampering(tmp_path: Path) -> None:
    path = tmp_path / "bank.npz"
    save_rollout_bank(path, _bank(), {"revision": 1})

    _rewrite(
        path,
        lambda payload: payload.__setitem__(
            "rollout_bank_id",
            np.asarray("0" * 64),
        ),
    )
    with pytest.raises(ValueError, match="ID does not match"):
        load_rollout_bank(path)

    save_rollout_bank(path, _bank(), {"revision": 1})
    _rewrite(path, lambda payload: payload.__setitem__("unexpected", np.asarray(1)))
    with pytest.raises(ValueError, match="archive members changed"):
        load_rollout_bank(path)


def test_rollout_bank_loader_accepts_legacy_archive(tmp_path: Path) -> None:
    bank = _bank()
    path = tmp_path / "legacy.npz"
    _legacy_archive(path, bank, {"legacy": True})

    restored, manifest = load_rollout_bank(path)

    assert restored.artifact_id == bank.artifact_id
    assert manifest == {"legacy": True}


def test_rollout_bank_io_rejects_nonfinite_manifest_before_publication(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.npz"
    with pytest.raises(ValueError, match="finite JSON"):
        save_rollout_bank(path, _bank(), {"invalid": float("nan")})
    assert not path.exists()


def test_rollout_bank_failed_validation_preserves_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "bank.npz"
    first = _bank()
    save_rollout_bank(path, first, {"revision": 1})
    original = path.read_bytes()

    def fail_validation(_path: Path):
        raise ValueError("synthetic validation failure")

    monkeypatch.setattr(rollout_bank_io, "load_rollout_bank", fail_validation)
    with pytest.raises(ValueError, match="synthetic validation failure"):
        save_rollout_bank(path, first, {"revision": 2})

    assert path.read_bytes() == original
