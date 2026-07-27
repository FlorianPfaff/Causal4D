"""Installed-wheel compatibility for the self-contained Prob4D provider-v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from three_repository_common import require
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
            "installed-wheel schema and independent consumer validation; not "
            "prospective covariance calibration or physical-prediction evidence"
        ),
    }
    return replace(
        artifact,
        source_revision=prob4d_revision,
        metadata=metadata,
    )


def _validate_bpt(path: Path) -> dict[str, object]:
    from bayesian_phystwin.observation_belief import load_observation_belief
    from bayesian_phystwin.prob4d_causal_lineage import (
        validate_claim_bearing_prob4d_observation_belief,
    )

    belief = load_observation_belief(path)
    result = validate_claim_bearing_prob4d_observation_belief(belief)
    provider = result.get("provider_attestation")
    require(isinstance(provider, dict), "BPT lost the provider-attestation summary")
    require(
        provider.get("claim_bearing") is True, "BPT did not require calibrated mode"
    )
    require(
        provider.get("runtime_revision_independently_verified") is True,
        "BPT did not require independently verified runtime provenance",
    )
    return result


def _validate_causal4d(path: Path) -> Any:
    from causal4d.claim_bearing_observation_lineage import (
        load_claim_bearing_prob4d_observation_lineage,
    )

    lineage = load_claim_bearing_prob4d_observation_lineage(path)
    provider = lineage.provider_validation.get("provider_attestation")
    require(
        isinstance(provider, dict),
        "Causal4D lost the provider-attestation summary",
    )
    require(
        provider.get("claim_bearing") is True,
        "Causal4D did not require calibrated mode",
    )
    require(
        provider.get("runtime_revision_independently_verified") is True,
        "Causal4D did not require independently verified runtime provenance",
    )
    return lineage


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

    bpt = _validate_bpt(observation_path)
    causal4d = _validate_causal4d(observation_path)
    producer = attested.metadata["prob4d_provider_attestation"]
    producer_manifest_id = producer["provider_manifest_id"]
    require(
        bpt["provider_attestation"]["provider_manifest_id"] == producer_manifest_id,
        "BPT validated a different provider manifest",
    )
    require(
        causal4d.provider_validation["provider_attestation"]["provider_manifest_id"]
        == producer_manifest_id,
        "Causal4D validated a different provider manifest",
    )

    tampered_metadata = deepcopy(dict(attested.metadata))
    tampered_metadata["prob4d_provider_attestation"]["provider_manifest"][
        "provider_version"
    ] = "tampered"
    tampered = replace(attested, metadata=tampered_metadata)
    tampered_path = output_dir / "rejected-provider-manifest-tamper.npz"
    save_observation_belief_export(tampered_path, tampered)
    rejections = [
        _expect_failure(
            "bpt:provider-manifest-tamper", lambda: _validate_bpt(tampered_path)
        ),
        _expect_failure(
            "causal4d:provider-manifest-tamper",
            lambda: _validate_causal4d(tampered_path),
        ),
    ]

    return {
        "schema": producer["schema_name"],
        "schema_version": producer["schema_version"],
        "prob4d_revision": prob4d_revision,
        "observation_artifact_id": attested.artifact_id,
        "provider_manifest_id": producer_manifest_id,
        "export_mode": producer["export_mode"],
        "covariance_root_mode": producer["covariance_root_mode"],
        "composition_jacobian_mode": producer["composition_jacobian_mode"],
        "runtime_revision": producer["runtime_revision"],
        "bpt_provider_attestation_validated": bpt["provider_attestation_validated"],
        "causal4d_provider_attestation_validated": (
            causal4d.provider_validation["provider_attestation_validated"]
        ),
        "fixture_only": True,
        "claim_evidence": False,
        "rejections": rejections,
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
