"""Verified admission boundary for contact-posterior diagnostics.

The topology-aware analyzer is deliberately reusable as a low-level numerical
kernel. Evidence-bearing callers must enter through this module so that exact
bundle integrity and row identities are validated before recomputation, while
host-local paths are excluded from the published diagnostic provenance.
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


def analyze_admitted_contact_posterior_bundle(
    bundle_directory: str | Path,
    *,
    config: DiagnosticConfig | None = None,
) -> dict[str, Any]:
    """Verify a source bundle, run the analyzer, and publish portable provenance."""

    source_integrity = verify_contact_posterior_source_bundle(bundle_directory)
    result = analyze_contact_posterior_bundle(
        bundle_directory,
        config=config,
    )
    if not isinstance(result, dict):
        raise TypeError("contact-posterior analyzer must return a dictionary")

    source_bundle = result.get("source_bundle")
    if not isinstance(source_bundle, dict):
        raise ValueError("diagnostic result is missing source-bundle provenance")
    analyzer_manifest = source_bundle.get("manifest_sha256")
    verified_manifest = source_integrity.get("manifest_sha256")
    if analyzer_manifest != verified_manifest:
        raise ValueError(
            "analyzer and admission verifier disagree on the source manifest"
        )

    portable_source = {
        key: value for key, value in source_bundle.items() if key != "directory"
    }
    portable_source["bundle_name"] = Path(bundle_directory).resolve().name
    portable_source["integrity_verification"] = source_integrity

    admitted = dict(result)
    admitted["source_bundle"] = portable_source
    admitted["admission_boundary"] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactPosteriorAdmission",
        "passed": True,
        "source_manifest_sha256": verified_manifest,
        "host_local_paths_published": False,
        "low_level_analyzer_verified_before_use": True,
    }
    return admitted
