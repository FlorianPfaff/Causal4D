from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

import causal4d.prob4d_observation_lineage as lineage
from causal4d.claim_bearing_observation_lineage import (
    require_claim_bearing_prob4d_lineage,
)
from causal4d.prob4d_provider_attestation import (
    compute_prob4d_provider_manifest_id,
)


def _semantic_result(value: str) -> dict[str, object]:
    return {
        "validated": True,
        "covariance_semantics": value,
        "cross_window_covariance_preserved": (
            value == lineage.PROB4D_JOINT_GAUGE_MODEL
        ),
    }


def _provider_manifest() -> dict[str, object]:
    descriptor: dict[str, object] = {
        "provider_name": "prob4d",
        "provider_version": "0.2.0",
        "provider_revision": "a" * 40,
        "provider_api_version": 2,
        "capabilities": [
            "analytic_sim3_composition_jacobians",
            "canonical_repeated_eigenspace_covariance_root",
            "explicit_exploratory_and_claim_bearing_exports",
            "provider_attested_observation_artifacts",
            "runtime_revision_attestation",
            "strict_prediction_calibration_compatibility",
        ],
        "artifact_schema_versions": {
            "ObservationBeliefV1": 1,
            "Prob4DCausalObservationStream": 2,
        },
        "limitations": {
            "uncalibrated_export_is_default": False,
            "deployment_environment_revision_is_independent_vcs_evidence": False,
        },
        "metadata": {
            "source_repository": "FlorianPfaff/Prob4D",
            "python_import_boundary": "prob4d.provider_v2",
        },
    }
    manifest_id = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {"manifest_id": manifest_id, **descriptor}


def _provider_attestation() -> dict[str, object]:
    manifest = _provider_manifest()
    return {
        "schema_name": "prob4d.provider-attestation",
        "schema_version": 1,
        "provider_api_version": 2,
        "provider_manifest_id": manifest["manifest_id"],
        "provider_manifest": manifest,
        "provider_revision": "a" * 40,
        "python_import_boundary": "prob4d.provider_v2",
        "export_mode": "calibrated",
        "claim_bearing": True,
        "calibration_compatibility_validated": True,
        "calibration_artifact_ids": {
            "gauge_artifact_id": "1" * 64,
            "point_artifact_id": "2" * 64,
        },
        "covariance_root_mode": "canonical_eigenspaces",
        "composition_jacobian_mode": "analytic",
        "runtime_revision": {
            "expected_revision": "a" * 40,
            "observed_revision": "a" * 40,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    }


def _claim_bearing_metadata() -> dict[str, object]:
    return {
        "prob4d_causal_stream_contract_version": 2,
        "prob4d_provider_attestation": _provider_attestation(),
        "covariance_calibration": {
            "status": "calibrated",
            "gauge_artifact_id": "1" * 64,
            "point_artifact_id": "2" * 64,
            "alignment_count": 2,
            "gauge_calibrated_alignment_count": 2,
            "covariance_fallback_counts": {},
            "uncalibrated_exploratory_covariance_allowed": False,
            "pointwise_covariance_fallback_allowed": False,
        },
    }


def _claim_bearing_descriptor() -> dict[str, object]:
    return {
        "source_revision": "a" * 40,
        "metadata": _claim_bearing_metadata(),
    }


def test_legacy_stream_version_is_inferred(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(
            lineage.PROB4D_LEGACY_COVARIANCE_SEMANTICS
        ),
    )

    result = lineage.validate_prob4d_causal_observation_metadata(
        {"metadata": {}},
        {},
    )

    assert result["stream_contract_version"] == 1
    assert result["stream_contract_version_inferred"] is True
    assert result["strict_causal_stream_contract"] is True
    assert result["provider_attestation_present"] is False


def test_explicit_joint_stream_version_is_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )

    result = lineage.validate_prob4d_causal_observation_metadata(
        {"metadata": {"prob4d_causal_stream_contract_version": 2}},
        {},
    )

    assert result["stream_contract_version"] == 2
    assert result["stream_contract_version_inferred"] is False
    assert result["strict_causal_stream_contract"] is True


def test_claim_bearing_provider_attestation_is_validated_independently(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )
    result = lineage.validate_claim_bearing_prob4d_observation_metadata(
        _claim_bearing_descriptor(),
        {},
    )

    provider = result["provider_attestation"]
    calibration = result["claim_bearing_covariance_calibration"]
    assert result["provider_attestation_present"] is True
    assert result["provider_attestation_validated"] is True
    assert result["claim_bearing_provider_v2_validated"] is True
    assert provider["provider_api_version"] == 2
    assert provider["claim_bearing"] is True
    assert provider["runtime_revision_independently_verified"] is True
    assert calibration["calibration_artifact_ids"] == {
        "gauge_artifact_id": "1" * 64,
        "point_artifact_id": "2" * 64,
    }
    assert calibration["covariance_fallback_counts"] == {}


def test_strict_provider_validation_rejects_provider_v1_artifact(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )
    with pytest.raises(ValueError, match="provider-v2 attestation is required"):
        lineage.validate_claim_bearing_prob4d_observation_metadata(
            {
                "source_revision": "a" * 40,
                "metadata": {"prob4d_causal_stream_contract_version": 2},
            },
            {},
        )


def test_provider_manifest_tampering_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )
    attestation = _provider_attestation()
    attestation["provider_manifest"]["provider_version"] = "999"
    with pytest.raises(ValueError, match="manifest ID does not match"):
        lineage.validate_prob4d_causal_observation_metadata(
            {
                "source_revision": "a" * 40,
                "metadata": {
                    "prob4d_causal_stream_contract_version": 2,
                    "prob4d_provider_attestation": attestation,
                },
            },
            {},
        )


def test_rehashed_provider_capability_removal_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )
    attestation = copy.deepcopy(_provider_attestation())
    manifest = attestation["provider_manifest"]
    manifest["capabilities"].remove("runtime_revision_attestation")
    manifest["manifest_id"] = compute_prob4d_provider_manifest_id(manifest)
    attestation["provider_manifest_id"] = manifest["manifest_id"]
    with pytest.raises(ValueError, match="required claim-bearing capabilities"):
        lineage.validate_prob4d_causal_observation_metadata(
            {
                "source_revision": "a" * 40,
                "metadata": {
                    "prob4d_causal_stream_contract_version": 2,
                    "prob4d_provider_attestation": attestation,
                },
            },
            {},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata["covariance_calibration"].__setitem__(
                "gauge_artifact_id", "3" * 64
            ),
            "gauge calibration metadata differs",
        ),
        (
            lambda metadata: metadata["covariance_calibration"].__setitem__(
                "gauge_calibrated_alignment_count", 1
            ),
            "uncalibrated gauge alignments",
        ),
        (
            lambda metadata: metadata["covariance_calibration"].__setitem__(
                "covariance_fallback_counts", {"pointwise": 1}
            ),
            "reports covariance fallback use",
        ),
        (
            lambda metadata: metadata["covariance_calibration"].__setitem__(
                "pointwise_covariance_fallback_allowed", True
            ),
            "cannot allow pointwise covariance fallback",
        ),
    ],
)
def test_claim_bearing_calibration_failures_are_rejected(
    monkeypatch,
    mutation,
    message: str,
) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )
    descriptor = _claim_bearing_descriptor()
    mutation(descriptor["metadata"])
    with pytest.raises(ValueError, match=message):
        lineage.validate_prob4d_causal_observation_metadata(descriptor, {})


def test_claim_bearing_requires_explicit_stream_v2(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )
    descriptor = _claim_bearing_descriptor()
    descriptor["metadata"].pop("prob4d_causal_stream_contract_version")
    with pytest.raises(ValueError, match="requires explicit causal stream contract"):
        lineage.validate_claim_bearing_prob4d_observation_metadata(descriptor, {})


def test_claim_bearing_lineage_wrapper_requires_validated_summary() -> None:
    provider = {
        "claim_bearing": True,
        "calibration_compatibility_validated": True,
        "runtime_revision_independently_verified": True,
    }
    calibration = {
        "status": "calibrated",
        "covariance_fallback_counts": {},
    }
    candidate = SimpleNamespace(
        provider_validation={
            "claim_bearing_provider_v2_validated": True,
            "strict_causal_stream_contract": True,
            "stream_contract_version": 2,
            "stream_contract_version_inferred": False,
            "covariance_semantics": lineage.PROB4D_JOINT_GAUGE_MODEL,
            "cross_window_covariance_preserved": True,
            "provider_attestation": provider,
            "claim_bearing_covariance_calibration": calibration,
        }
    )
    assert require_claim_bearing_prob4d_lineage(candidate) is candidate

    incomplete = copy.deepcopy(candidate.provider_validation)
    incomplete["claim_bearing_provider_v2_validated"] = False
    with pytest.raises(ValueError, match="complete claim-bearing"):
        require_claim_bearing_prob4d_lineage(
            SimpleNamespace(provider_validation=incomplete)
        )

    missing_calibration = copy.deepcopy(candidate.provider_validation)
    missing_calibration.pop("claim_bearing_covariance_calibration")
    with pytest.raises(ValueError, match="covariance calibration"):
        require_claim_bearing_prob4d_lineage(
            SimpleNamespace(provider_validation=missing_calibration)
        )


def test_joint_stream_rejects_mismatched_explicit_version(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(lineage.PROB4D_JOINT_GAUGE_MODEL),
    )

    with pytest.raises(ValueError, match="disagrees with covariance semantics"):
        lineage.validate_prob4d_causal_observation_metadata(
            {"metadata": {"prob4d_causal_stream_contract_version": 1}},
            {},
        )


def test_fixed_lag_is_not_a_strict_stream_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(
            lineage.PROB4D_FIXED_LAG_GAUGE_MODEL
        ),
    )

    result = lineage.validate_prob4d_causal_observation_metadata(
        {"metadata": {}},
        {},
    )

    assert result["stream_contract_version"] is None
    assert result["stream_contract_version_inferred"] is False
    assert result["strict_causal_stream_contract"] is False


def test_fixed_lag_rejects_an_explicit_strict_version(monkeypatch) -> None:
    monkeypatch.setattr(
        lineage,
        "_validate_prob4d_semantics",
        lambda descriptor, arrays: _semantic_result(
            lineage.PROB4D_FIXED_LAG_GAUGE_MODEL
        ),
    )

    with pytest.raises(ValueError, match="fixed-lag covariance cannot declare"):
        lineage.validate_prob4d_causal_observation_metadata(
            {"metadata": {"prob4d_causal_stream_contract_version": 2}},
            {},
        )
