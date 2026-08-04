import pytest

from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
    PhysicalBeliefProviderManifest,
    validate_provider_compatibility,
)


def test_falsey_non_mapping_artifact_requirements_are_rejected() -> None:
    manifest = PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="a" * 40,
        schema_version=1,
        capabilities=BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
        artifact_schema_versions={"GraphBelief": 1, "TwinBelief": 1},
    )

    with pytest.raises(ValueError, match="required_artifact_versions must be a mapping"):
        validate_provider_compatibility(
            manifest,
            required_artifact_versions=[],  # type: ignore[arg-type]
        )
