"""Public facade for result-bundle identity and reproduction contracts."""

from result_bundle_identity import (
    RESULT_MANIFEST_NAME,
    ArtifactRecord,
    VerifiedResultBundle,
    load_strict_json,
    load_strict_json_bytes,
    sha256_file,
    verify_result_manifest,
)
from reproduction_manifest_runtime import (
    COMPARISON_CONTRACT_VERSION,
    REPRODUCTION_MANIFEST_KIND,
    REPRODUCTION_MANIFEST_SCHEMA_VERSION,
    build_reproduction_manifest,
    capture_runtime_identity,
    write_reproduction_manifest,
)
from reproduction_manifest_validation import validate_reproduction_manifest

__all__ = [
    "RESULT_MANIFEST_NAME",
    "REPRODUCTION_MANIFEST_SCHEMA_VERSION",
    "COMPARISON_CONTRACT_VERSION",
    "REPRODUCTION_MANIFEST_KIND",
    "ArtifactRecord",
    "VerifiedResultBundle",
    "sha256_file",
    "load_strict_json_bytes",
    "load_strict_json",
    "verify_result_manifest",
    "capture_runtime_identity",
    "build_reproduction_manifest",
    "write_reproduction_manifest",
    "validate_reproduction_manifest",
]
