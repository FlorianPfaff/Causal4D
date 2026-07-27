from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.observation_contract_bundle import (
    OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256,
    invalid_observation_contract_vector,
    observation_contract_artifact_id,
    observation_contract_bundle_manifest,
    observation_contract_invalid_cases,
    observation_contract_schema,
    observation_contract_vector,
)
from causal4d.observation_lineage import (
    compute_observation_artifact_id,
    load_observation_lineage,
)


def _write(
    path: Path,
    descriptor,
    arrays,
    *,
    artifact_id: str | None = None,
) -> None:
    payload = dict(descriptor)
    payload["artifact_id"] = (
        observation_contract_artifact_id(payload, arrays)
        if artifact_id is None
        else artifact_id
    )
    np.savez_compressed(
        path,
        descriptor_json=np.asarray(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        **arrays,
    )


def test_bundle_is_content_locked_and_normative() -> None:
    manifest = observation_contract_bundle_manifest()
    schema = observation_contract_schema()

    assert manifest["bundle_sha256"] == OBSERVATION_BELIEF_CONTRACT_BUNDLE_SHA256
    assert manifest["canonical_repository"] == "FlorianPfaff/Prob4D"
    assert schema["descriptor"]["closed"] is True
    assert schema["arrays"]["closed"] is True


@pytest.mark.parametrize("vector_name", ("minimal", "zero_rank"))
def test_causal4d_accepts_every_valid_vector(
    vector_name: str,
    tmp_path: Path,
) -> None:
    vector = observation_contract_vector(vector_name)
    assert (
        compute_observation_artifact_id(vector.descriptor, vector.arrays)
        == vector.expected_artifact_id
    )

    path = tmp_path / f"{vector_name}.npz"
    _write(path, vector.descriptor, vector.arrays)
    lineage = load_observation_lineage(path)
    assert lineage.artifact_id == vector.expected_artifact_id
    assert lineage.factor_rank == len(vector.descriptor["factor_names"])


@pytest.mark.parametrize(
    "case_id",
    [case["id"] for case in observation_contract_invalid_cases()],
)
def test_causal4d_rejects_every_invalid_vector(
    case_id: str,
    tmp_path: Path,
) -> None:
    invalid = invalid_observation_contract_vector(case_id)
    artifact_id = (
        invalid.original_artifact_id
        if invalid.mode == "digest_mismatch"
        else observation_contract_artifact_id(
            invalid.descriptor,
            invalid.arrays,
        )
    )
    path = tmp_path / f"{case_id}.npz"
    _write(
        path,
        invalid.descriptor,
        invalid.arrays,
        artifact_id=artifact_id,
    )
    with pytest.raises(ValueError):
        load_observation_lineage(path)
