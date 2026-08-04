"""Verified admission boundary for contact-posterior diagnostics.

The topology-aware analyzer is deliberately reusable as a low-level numerical
kernel. Evidence-bearing callers must enter through this module so exact byte
identity, domain-specific row contracts, and portable provenance are established
before any interpretation is published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from causal4d.contact_posterior_diagnostics import (
    DiagnosticConfig,
    analyze_contact_posterior_bundle,
)
from causal4d.contact_posterior_source_integrity import (
    verify_contact_posterior_source_bundle,
)
from causal4d.result_bundle_verification import verify_embedded_result_bundle


def _verify_source_bundle(
    bundle_directory: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    embedded = verify_embedded_result_bundle(bundle_directory)
    domain = verify_contact_posterior_source_bundle(bundle_directory)
    if embedded.get("manifest_sha256") != domain.get("manifest_sha256"):
        raise ValueError("source-integrity verifiers disagree on manifest identity")
    return embedded, domain


def analyze_admitted_contact_posterior_bundle(
    bundle_directory: str | Path,
    *,
    config: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Verify a source bundle, run the analyzer, and publish portable provenance."""

    embedded_integrity, source_integrity = _verify_source_bundle(bundle_directory)
    embedded_manifest = embedded_integrity.get("manifest_sha256")

    result = analyze_contact_posterior_bundle(
        bundle_directory,
        config=config,
    )
    if not isinstance(result, dict):
        raise TypeError("contact-posterior analyzer must return a dictionary")

    embedded_after, source_after = _verify_source_bundle(bundle_directory)
    if (
        embedded_after != embedded_integrity
        or source_after != source_integrity
    ):
        raise ValueError(
            "source bundle changed during contact-posterior analysis"
        )

    source_bundle = result.get("source_bundle")
    if not isinstance(source_bundle, dict):
        raise ValueError("diagnostic result is missing source-bundle provenance")
    analyzer_manifest = source_bundle.get("manifest_sha256")
    if analyzer_manifest != embedded_manifest:
        raise ValueError(
            "analyzer and admission verifiers disagree on the source manifest"
        )

    combined_integrity = dict(source_integrity)
    combined_integrity["embedded_bundle"] = embedded_integrity
    portable_source = {
        key: value for key, value in source_bundle.items() if key != "directory"
    }
    portable_source.update(
        {
            "bundle_name": embedded_integrity["bundle_name"],
            "manifest_sha256": embedded_manifest,
            "artifacts": embedded_integrity["artifacts"],
            "integrity_verification": combined_integrity,
        }
    )

    admitted = dict(result)
    admitted["source_bundle"] = portable_source
    admitted["admission_boundary"] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactPosteriorAdmission",
        "passed": True,
        "source_manifest_sha256": embedded_manifest,
        "host_local_paths_published": False,
        "byte_identity_verified": True,
        "domain_row_contracts_verified": True,
        "low_level_analyzer_verified_before_use": True,
        "source_stable_through_analysis": True,
        "bundle_name": Path(bundle_directory).resolve().name,
    }
    return admitted
