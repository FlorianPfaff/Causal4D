import pytest

from causal4d.execution_block_provenance import (
    bind_execution_block_source_manifest,
    extract_execution_block_manifest_binding,
    validate_execution_block_target_manifest,
)


def _manifest(*, freeze: bool = True) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "outer_fold_id": "hold-left_forepaw-lift_high",
        "protocol_id": "causal4d-sloth-multi-action-v1",
        "protocol_design_sha256": "1" * 64,
        "preacquisition_plan_id": "causal4d-sloth-preacquisition-v4",
        "preacquisition_amendment_sha256": "2" * 64,
    }
    if freeze:
        result["method_freeze_sha256"] = "3" * 64
    return result


def _source_metadata(*, freeze: bool = True) -> dict[str, object]:
    return bind_execution_block_source_manifest(
        _manifest(freeze=freeze),
        manifest_sha256="4" * 64,
    )


def test_source_binding_requires_complete_registered_identity() -> None:
    manifest = _manifest()
    del manifest["protocol_design_sha256"]
    with pytest.raises(ValueError, match="protocol_design_sha256"):
        extract_execution_block_manifest_binding(manifest)


def test_source_binding_rejects_noncanonical_digest() -> None:
    manifest = _manifest()
    manifest["preacquisition_amendment_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        bind_execution_block_source_manifest(
            manifest,
            manifest_sha256="4" * 64,
        )


def test_matching_source_and_target_binding_is_canonical() -> None:
    binding = validate_execution_block_target_manifest(
        _source_metadata(),
        _manifest(),
        expected_outer_fold_id="hold-left_forepaw-lift_high",
        target_manifest_sha256="5" * 64,
    )
    assert binding["verified"]
    assert binding["protocol_design_sha256"] == "1" * 64
    assert binding["method_freeze_sha256"] == "3" * 64
    assert binding["source_manifest_sha256"] == "4" * 64
    assert binding["target_manifest_sha256"] == "5" * 64


def test_target_protocol_digest_mismatch_is_rejected() -> None:
    target = _manifest()
    target["protocol_design_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="protocol_design_sha256"):
        validate_execution_block_target_manifest(
            _source_metadata(),
            target,
            expected_outer_fold_id="hold-left_forepaw-lift_high",
            target_manifest_sha256="5" * 64,
        )


def test_target_amendment_mismatch_is_rejected() -> None:
    target = _manifest()
    target["preacquisition_amendment_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="preacquisition_amendment_sha256"):
        validate_execution_block_target_manifest(
            _source_metadata(),
            target,
            expected_outer_fold_id="hold-left_forepaw-lift_high",
            target_manifest_sha256="5" * 64,
        )


def test_method_freeze_must_be_bound_symmetrically() -> None:
    with pytest.raises(ValueError, match="present in both"):
        validate_execution_block_target_manifest(
            _source_metadata(freeze=False),
            _manifest(freeze=True),
            expected_outer_fold_id="hold-left_forepaw-lift_high",
            target_manifest_sha256="5" * 64,
        )


def test_target_fold_mismatch_is_rejected() -> None:
    target = _manifest()
    target["outer_fold_id"] = "another-fold"
    with pytest.raises(ValueError, match="outer_fold_id"):
        validate_execution_block_target_manifest(
            _source_metadata(),
            target,
            expected_outer_fold_id="hold-left_forepaw-lift_high",
            target_manifest_sha256="5" * 64,
        )


def test_source_manifest_digest_is_required_for_target_evaluation() -> None:
    metadata = _source_metadata()
    del metadata["source_manifest_sha256"]
    with pytest.raises(ValueError, match="source_manifest_sha256"):
        validate_execution_block_target_manifest(
            metadata,
            _manifest(),
            expected_outer_fold_id="hold-left_forepaw-lift_high",
            target_manifest_sha256="5" * 64,
        )
