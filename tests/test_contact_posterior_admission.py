from __future__ import annotations

from pathlib import Path

import pytest

import causal4d.contact_posterior_admission as admission
from causal4d.contact_posterior_diagnostics import DiagnosticConfig


_MANIFEST = "a" * 64


def _embedded_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "causal4d-latent-contact-v1",
        "bundle_name": "bundle",
        "manifest_sha256": _MANIFEST,
        "artifact_count": 6,
        "artifacts": {
            "summary.json": {"bytes": 2, "sha256": "b" * 64},
        },
    }


def _source_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "causal4d-latent-contact-v1",
        "manifest_sha256": _MANIFEST,
        "artifact_count": 6,
        "seed_count": 20,
        "contact_recovery_row_count": 120,
        "intervention_row_count": 720,
        "online_case_count": 120,
        "passed": True,
    }


def _analysis_result() -> dict[str, object]:
    return {
        "source_bundle": {
            "directory": "/runner/work/Causal4D/outputs/bundle",
            "manifest_sha256": _MANIFEST,
            "benchmark": "causal4d-latent-contact-v1",
            "seeds": [100, 101],
        },
        "recomputation_parity": {"passed": True},
        "overall": {},
        "by_topology": [],
        "claim_boundary": "controlled diagnostic",
    }


def test_admission_verifies_before_analysis_and_removes_host_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    calls: list[str] = []
    config = DiagnosticConfig()

    def verify_embedded(path: str | Path) -> dict[str, object]:
        assert Path(path) == bundle
        calls.append("embedded")
        return _embedded_report()

    def verify_source(path: str | Path) -> dict[str, object]:
        assert Path(path) == bundle
        calls.append("source")
        return _source_report()

    def analyze(
        path: str | Path,
        *,
        config: DiagnosticConfig | None = None,
    ) -> dict[str, object]:
        assert Path(path) == bundle
        assert config is not None
        calls.append("analyze")
        return _analysis_result()

    monkeypatch.setattr(admission, "verify_embedded_result_bundle", verify_embedded)
    monkeypatch.setattr(
        admission,
        "verify_contact_posterior_source_bundle",
        verify_source,
    )
    monkeypatch.setattr(admission, "analyze_contact_posterior_bundle", analyze)

    result = admission.analyze_admitted_contact_posterior_bundle(
        bundle,
        config=config,
    )

    assert calls == ["embedded", "source", "analyze"]
    source_bundle = result["source_bundle"]
    assert isinstance(source_bundle, dict)
    assert "directory" not in source_bundle
    assert source_bundle["bundle_name"] == "bundle"
    assert source_bundle["manifest_sha256"] == _MANIFEST
    assert source_bundle["artifacts"] == _embedded_report()["artifacts"]
    integrity = source_bundle["integrity_verification"]
    assert isinstance(integrity, dict)
    assert integrity["embedded_bundle"] == _embedded_report()
    assert result["admission_boundary"] == {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactPosteriorAdmission",
        "passed": True,
        "source_manifest_sha256": _MANIFEST,
        "host_local_paths_published": False,
        "byte_identity_verified": True,
        "domain_row_contracts_verified": True,
        "low_level_analyzer_verified_before_use": True,
        "bundle_name": "bundle",
    }


def test_integrity_failure_prevents_analyzer_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analyzed = False

    def reject(_: str | Path) -> dict[str, object]:
        raise ValueError("tampered source bundle")

    def analyze(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal analyzed
        analyzed = True
        return _analysis_result()

    monkeypatch.setattr(admission, "verify_embedded_result_bundle", reject)
    monkeypatch.setattr(admission, "analyze_contact_posterior_bundle", analyze)

    with pytest.raises(ValueError, match="tampered source bundle"):
        admission.analyze_admitted_contact_posterior_bundle(tmp_path)
    assert analyzed is False


def test_verifier_manifest_disagreement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source_report()
    source["manifest_sha256"] = "c" * 64
    monkeypatch.setattr(
        admission,
        "verify_embedded_result_bundle",
        lambda _: _embedded_report(),
    )
    monkeypatch.setattr(
        admission,
        "verify_contact_posterior_source_bundle",
        lambda _: source,
    )

    with pytest.raises(ValueError, match="verifiers disagree"):
        admission.analyze_admitted_contact_posterior_bundle(tmp_path)


def test_analyzer_manifest_disagreement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _analysis_result()
    source_bundle = result["source_bundle"]
    assert isinstance(source_bundle, dict)
    source_bundle["manifest_sha256"] = "d" * 64
    monkeypatch.setattr(
        admission,
        "verify_embedded_result_bundle",
        lambda _: _embedded_report(),
    )
    monkeypatch.setattr(
        admission,
        "verify_contact_posterior_source_bundle",
        lambda _: _source_report(),
    )
    monkeypatch.setattr(
        admission,
        "analyze_contact_posterior_bundle",
        lambda *args, **kwargs: result,
    )

    with pytest.raises(ValueError, match="analyzer and admission"):
        admission.analyze_admitted_contact_posterior_bundle(
            tmp_path
        )