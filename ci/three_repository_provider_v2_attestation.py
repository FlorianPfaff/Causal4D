"""Installed-wheel compatibility for the strict Prob4D provider-v2 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from three_repository_common import array_digest, require
from three_repository_observation import fixture_artifact


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _expect_failure(label: str, operation: Callable[[], Any]) -> dict[str, str]:
    try:
        operation()
    except (RuntimeError, ValueError) as error:
        return {
            "label": label,
            "error": type(error).__name__,
            "message": str(error),
        }
    raise RuntimeError(f"rejection case {label!r} was incorrectly accepted")


def _build_attested_fixture(artifact: Any, *, prob4d_revision: str) -> Any:
    from prob4d.provider_v2 import (
        RuntimeRevisionAttestation,
        build_provider_attestation,
        prob4d_provider_manifest,
    )

    runtime = RuntimeRevisionAttestation(
        expected_revision=prob4d_revision,
        observed_revision=prob4d_revision,
        source="source_checkout",
        clean_checkout=True,
        matched=True,
        independently_verified=True,
    )
    calibration_ids = {
        "gauge_artifact_id": _digest("contract-test-gauge-calibration-v1"),
        "point_artifact_id": _digest("contract-test-point-calibration-v1"),
    }
    metadata = deepcopy(dict(artifact.metadata))
    posterior = metadata.get("gauge_posterior")
    require(isinstance(posterior, dict), "fixture gauge posterior is missing")
    alignments = posterior.get("alignments")
    require(isinstance(alignments, list), "fixture gauge alignments are missing")
    alignment_count = len(alignments)

    metadata["prob4d_causal_stream_contract_version"] = 2
    metadata["prob4d_causal_stream_contract"] = {
        "version": 2,
        "causal_frame_stop_convention": "exclusive",
    }
    metadata["metric_anchor_covariance_in_joint_factor"] = True
    anchor = metadata.get("metric_gauge_anchor")
    require(isinstance(anchor, dict), "fixture metric anchor is missing")
    anchor.update(
        {
            "schema_name": "prob4d.metric-gauge-anchor",
            "schema_version": 1,
            "case_id": artifact.case_id,
            "coordinate_frame": metadata["coordinate_frame"],
            "world_frame_id": metadata["coordinate_frame"],
            "metric_units": "m",
            "calibration_artifact_sha256": _digest(
                "contract-test-metric-anchor-calibration-v1"
            ),
            "covariance_treatment": "propagated_external_prior",
        }
    )
    metadata["covariance_calibration"] = {
        "status": "calibrated",
        **calibration_ids,
        "alignment_count": alignment_count,
        "gauge_calibrated_alignment_count": alignment_count,
        "covariance_fallback_counts": {},
        "uncalibrated_exploratory_covariance_allowed": False,
        "pointwise_covariance_fallback_allowed": False,
    }
    metadata["prob4d_provider_attestation"] = build_provider_attestation(
        provider_manifest=prob4d_provider_manifest(
            provider_revision=prob4d_revision,
        ),
        provider_revision=prob4d_revision,
        export_mode="calibrated",
        calibration_compatibility_validated=True,
        calibration_artifact_ids=calibration_ids,
        covariance_root_mode="canonical_eigenspaces",
        composition_jacobian_mode="analytic",
        runtime_revision=runtime.as_metadata(),
    )
    metadata["provider_attestation_contract_test"] = {
        "claim_evidence": False,
        "fixture_only": True,
        "synthetic_calibration_ids": True,
        "purpose": (
            "installed-wheel schema, strict update admission, and independent "
            "consumer validation; not prospective covariance calibration or "
            "physical-prediction evidence"
        ),
    }
    return replace(
        artifact,
        source_revision=prob4d_revision,
        metadata=metadata,
    )


def _validate_prob4d(path: Path) -> dict[str, object]:
    from prob4d.provider_v2_loading import load_claim_bearing_observation_belief

    validated = load_claim_bearing_observation_belief(path)
    return {
        "artifact_id": validated.artifact_id,
        "provider_manifest_id": validated.provider_manifest_id,
        "gauge_calibration_id": validated.gauge_calibration_id,
        "point_calibration_id": validated.point_calibration_id,
        "runtime_revision": validated.runtime_revision,
    }


def _state_design(observation_count: int) -> np.ndarray:
    require(observation_count == 6, "claim-bearing fixture observation count changed")
    design = np.zeros((observation_count, 3, 1), dtype=np.float64)
    design[0, 1, 0] = 1.0
    design[0, 2, 0] = -1.0
    design[1, 0, 0] = 1.0
    design[2, 1, 0] = -1.0
    design[3, 2, 0] = 1.0
    design[5, 0, 0] = -1.0
    return design


def _validate_bpt(path: Path) -> dict[str, object]:
    from bayesian_phystwin.claim_bearing_prob4d import (
        build_claim_bearing_gauge_aware_batch_from_observation_belief,
    )
    from bayesian_phystwin.gauge_aware_belief import (
        GaugeAwareBeliefConfig,
        update_gauge_aware_belief,
    )
    from bayesian_phystwin.observation_belief import load_observation_belief
    from bayesian_phystwin.prob4d_causal_lineage import (
        validate_claim_bearing_prob4d_observation_belief,
    )

    belief = load_observation_belief(path)
    validation = validate_claim_bearing_prob4d_observation_belief(belief)
    state = _state_design(belief.observation_count)
    injected_coefficient_m = 0.004
    prediction = belief.mean_xyz_m - injected_coefficient_m * state[:, :, 0]
    query = np.zeros((1, 3, 1), dtype=np.float64)
    query[0, 0, 0] = 1.0
    adapted = build_claim_bearing_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=prediction,
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_scale_m=0.02,
        state_prior_covariance_m2=np.asarray([[4e-4]], dtype=np.float64),
    )
    metadata = adapted.batch.metadata
    require(isinstance(metadata, Mapping), "BPT adapted batch metadata is missing")
    require(
        metadata.get("prob4d_claim_bearing_provider_v2_validated") is True,
        "BPT strict adapter did not retain claim-bearing validation",
    )
    provider = validation.get("provider_attestation")
    require(isinstance(provider, Mapping), "BPT lost the provider-attestation summary")
    require(
        metadata.get("prob4d_claim_bearing_provider_manifest_id")
        == provider.get("provider_manifest_id"),
        "BPT strict adapter bound a different provider manifest",
    )

    config = GaugeAwareBeliefConfig(maximum_iterations=20)
    first = update_gauge_aware_belief(adapted.batch, config=config)
    second = update_gauge_aware_belief(adapted.batch, config=config)
    require(first.accepted, f"BPT claim-bearing update abstained: {first.reason}")
    require(second.accepted, f"repeated BPT update abstained: {second.reason}")
    np.testing.assert_array_equal(first.state_coefficients, second.state_coefficients)
    np.testing.assert_array_equal(
        first.posterior_covariance,
        second.posterior_covariance,
    )
    coefficient = float(first.state_coefficients[0])
    require(
        0.002 <= coefficient <= 0.006,
        f"BPT claim-bearing update left the deterministic interval: {coefficient}",
    )
    return {
        "claim_bearing_provider_v2_validated": validation[
            "claim_bearing_provider_v2_validated"
        ],
        "provider_manifest_id": provider["provider_manifest_id"],
        "calibration": validation["claim_bearing_covariance_calibration"],
        "state_coefficient_m": coefficient,
        "injected_coefficient_m": injected_coefficient_m,
        "update_id": array_digest(
            first.state_coefficients,
            first.posterior_covariance,
            first.robust_weights,
        ),
        "adapter": adapted.summary(),
    }


def _validate_causal4d(path: Path) -> dict[str, object]:
    from causal4d.claim_bearing_observation_lineage import (
        load_claim_bearing_prob4d_observation_lineage,
    )

    lineage = load_claim_bearing_prob4d_observation_lineage(path)
    validation = lineage.provider_validation
    provider = validation.get("provider_attestation")
    calibration = validation.get("claim_bearing_covariance_calibration")
    require(
        validation.get("claim_bearing_provider_v2_validated") is True,
        "Causal4D did not complete claim-bearing provider-v2 validation",
    )
    require(
        isinstance(provider, Mapping),
        "Causal4D lost the provider-attestation summary",
    )
    require(
        isinstance(calibration, Mapping),
        "Causal4D lost the covariance-calibration summary",
    )
    return {
        "artifact_id": lineage.artifact_id,
        "provider_manifest_id": provider["provider_manifest_id"],
        "calibration": calibration,
        "stream_contract_version": validation["stream_contract_version"],
        "stream_contract_version_inferred": validation[
            "stream_contract_version_inferred"
        ],
        "covariance_semantics": validation["covariance_semantics"],
    }


def _write_variant(
    artifact: Any,
    target: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    from prob4d.provider_v2 import save_observation_belief_export

    metadata = deepcopy(dict(artifact.metadata))
    mutate(metadata)
    save_observation_belief_export(target, replace(artifact, metadata=metadata))


def _consumer_rejections(path: Path, label: str) -> list[dict[str, str]]:
    return [
        _expect_failure(
            f"prob4d:{label}",
            lambda: _validate_prob4d(path),
        ),
        _expect_failure(
            f"bpt:{label}",
            lambda: _validate_bpt(path),
        ),
        _expect_failure(
            f"causal4d:{label}",
            lambda: _validate_causal4d(path),
        ),
    ]


def _rejection_corpus(artifact: Any, output_dir: Path) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    def provider_manifest_tamper(metadata: dict[str, Any]) -> None:
        metadata["prob4d_provider_attestation"]["provider_manifest"][
            "provider_version"
        ] = "tampered"

    path = output_dir / "rejected-provider-manifest-tamper.npz"
    _write_variant(artifact, path, provider_manifest_tamper)
    results.extend(_consumer_rejections(path, "provider-manifest-tamper"))

    def calibration_identity_drift(metadata: dict[str, Any]) -> None:
        metadata["covariance_calibration"]["gauge_artifact_id"] = "3" * 64

    path = output_dir / "rejected-calibration-identity-drift.npz"
    _write_variant(artifact, path, calibration_identity_drift)
    results.extend(_consumer_rejections(path, "calibration-identity-drift"))

    def incomplete_gauge_calibration(metadata: dict[str, Any]) -> None:
        metadata["covariance_calibration"]["gauge_calibrated_alignment_count"] -= 1

    path = output_dir / "rejected-incomplete-gauge-calibration.npz"
    _write_variant(artifact, path, incomplete_gauge_calibration)
    results.extend(_consumer_rejections(path, "incomplete-gauge-calibration"))

    def covariance_fallback(metadata: dict[str, Any]) -> None:
        metadata["covariance_calibration"]["covariance_fallback_counts"] = {
            "pointwise": 1
        }

    path = output_dir / "rejected-covariance-fallback.npz"
    _write_variant(artifact, path, covariance_fallback)
    results.extend(_consumer_rejections(path, "covariance-fallback"))

    def fallback_permission(metadata: dict[str, Any]) -> None:
        metadata["covariance_calibration"][
            "pointwise_covariance_fallback_allowed"
        ] = True

    path = output_dir / "rejected-fallback-permission.npz"
    _write_variant(artifact, path, fallback_permission)
    results.extend(_consumer_rejections(path, "fallback-permission"))

    def inferred_stream_version(metadata: dict[str, Any]) -> None:
        metadata.pop("prob4d_causal_stream_contract_version")

    path = output_dir / "rejected-inferred-stream-version.npz"
    _write_variant(artifact, path, inferred_stream_version)
    results.extend(_consumer_rejections(path, "inferred-stream-version"))
    return results


def run(
    *,
    fixture_path: Path,
    prob4d_revision: str,
    output_dir: Path,
) -> dict[str, object]:
    from prob4d.provider_v2 import save_observation_belief_export

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture, _ = fixture_artifact(fixture_path)
    attested = _build_attested_fixture(
        fixture,
        prob4d_revision=prob4d_revision,
    )
    observation_path = output_dir / "provider-v2-observation.npz"
    save_observation_belief_export(observation_path, attested)

    producer = _validate_prob4d(observation_path)
    bpt = _validate_bpt(observation_path)
    causal4d = _validate_causal4d(observation_path)
    provider_manifest_id = producer["provider_manifest_id"]
    require(
        bpt["provider_manifest_id"] == provider_manifest_id,
        "BPT validated a different provider manifest",
    )
    require(
        causal4d["provider_manifest_id"] == provider_manifest_id,
        "Causal4D validated a different provider manifest",
    )
    require(
        producer["artifact_id"] == bpt["adapter"]["source_observation_belief_id"],
        "BPT strict adapter lost the claim-bearing observation identity",
    )
    require(
        producer["artifact_id"] == causal4d["artifact_id"],
        "Causal4D validated a different observation artifact",
    )

    return {
        "status": "passed",
        "schema_version": 2,
        "prob4d_revision": prob4d_revision,
        "observation_artifact_id": attested.artifact_id,
        "provider_manifest_id": provider_manifest_id,
        "producer": producer,
        "bpt": bpt,
        "causal4d": causal4d,
        "fixture_only": True,
        "claim_evidence": False,
        "rejections": _rejection_corpus(attested, output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob4d-fixture", type=Path, required=True)
    parser.add_argument("--prob4d-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(
        fixture_path=args.prob4d_fixture,
        prob4d_revision=args.prob4d_revision,
        output_dir=args.output_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
