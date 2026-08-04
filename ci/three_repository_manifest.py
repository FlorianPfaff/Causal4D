"""Evidence-manifest checks for the installed-wheel golden path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from three_repository_common import (
    BAYESIAN_PHYSTWIN_REPOSITORY,
    CAUSAL4D_REPOSITORY,
    PROB4D_REPOSITORY,
    require,
)


def _expect_failure(label: str, operation: Callable[[], Any]) -> dict[str, str]:
    try:
        operation()
    except (RuntimeError, ValueError) as error:
        return {
            "label": label,
            "error": type(error).__name__,
            "message": str(error),
        }
    raise RuntimeError(f"evidence rejection case {label!r} was accepted")


def _artifact(path: Path, *, name: str, role: str) -> Any:
    from bayesian_phystwin.run_manifest import ArtifactDigest, sha256_file

    return ArtifactDigest(
        name=name,
        role=role,
        path=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _require_promotable(manifest: Any) -> None:
    require(not manifest.dirty, "promotable evidence manifest is dirty")
    require(bool(manifest.method_freeze_id), "method freeze ID is missing")
    require(bool(manifest.protocol_id), "protocol ID is missing")
    require(bool(manifest.split_id), "split ID is missing")
    require(bool(manifest.baseline_id), "baseline ID is missing")
    require(bool(manifest.inputs), "evidence manifest has no inputs")
    require(bool(manifest.outputs), "evidence manifest has no outputs")
    require(
        all(not repository.dirty for repository in manifest.related_repositories),
        "related repository state is dirty",
    )


def run_evidence_manifest_checks(
    *,
    workdir: Path,
    observation_path: Path,
    twin_belief_path: Path,
    rollout_bank_path: Path,
    revisions: dict[str, str],
    provider_manifest_id: str,
    observation_artifact_id: str,
) -> dict[str, Any]:
    """Write, reload, and negatively test one complete V2 run manifest."""

    from bayesian_phystwin.repository_provenance import RepositoryState
    from bayesian_phystwin.run_manifest import installed_package_versions
    from bayesian_phystwin.run_manifest_v2 import (
        RunManifestV2,
        load_run_manifest_v2,
        verify_run_manifest_artifacts,
        write_run_manifest,
    )

    manifest = RunManifestV2(
        run_id="three-repository-installed-wheel-golden-path",
        repository=CAUSAL4D_REPOSITORY,
        revision=revisions["causal4d"],
        dirty=False,
        related_repositories=(
            RepositoryState(
                repository=PROB4D_REPOSITORY,
                revision=revisions["prob4d"],
                dirty=False,
                role="observation",
            ),
            RepositoryState(
                repository=BAYESIAN_PHYSTWIN_REPOSITORY,
                revision=revisions["bayesian-phystwin"],
                dirty=False,
                role="upstream",
            ),
        ),
        command=("python", "three_repository_golden_path.py"),
        classification="infrastructure",
        statistical_unit="deterministic joint-gauge contract fixture",
        information_boundary={
            "causal_frame_stop_exclusive": 6,
            "future_prediction_payloads_opened": 0,
        },
        configuration={
            "observation_artifact_id": observation_artifact_id,
            "provider_manifest_id": provider_manifest_id,
            "prob4d_causal_stream_contract_version": 2,
            "replay_provider": "deterministic-cpu-fake-v2",
        },
        inputs=(
            _artifact(
                observation_path,
                name="prob4d-observation-belief",
                role="input",
            ),
        ),
        outputs=(
            _artifact(
                twin_belief_path,
                name="causal4d-twin-belief",
                role="output",
            ),
            _artifact(
                rollout_bank_path,
                name="causal4d-rollout-bank",
                role="output",
            ),
        ),
        package_versions=installed_package_versions(
            ("prob4d", "bayesian-phystwin", "causal4d", "numpy")
        ),
        runtime_environment={
            "execution_boundary": "clean-installed-wheel-virtual-environment",
            "pythonpath_unset": True,
        },
        method_freeze_id="three-repository-golden-path-v1",
        protocol_id="prob4d-bpt-causal4d-installed-wheel-v1",
        split_id="deterministic-contract-fixture-v1",
        baseline_id="fake-replay-provider-v2",
        created_utc="2026-07-27T00:00:00+00:00",
        notes="No empirical performance claim; this is an interoperability check.",
    )
    _require_promotable(manifest)
    manifest_path = workdir / "run-manifest-v2.json"
    write_run_manifest(manifest_path, manifest)
    restored = load_run_manifest_v2(manifest_path)
    _require_promotable(restored)
    require(restored.manifest_id == manifest.manifest_id, "manifest ID changed")
    require(
        restored.evidence_fingerprint == manifest.evidence_fingerprint,
        "evidence fingerprint changed",
    )
    verify_run_manifest_artifacts(restored, root=workdir)

    incomplete_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    incomplete_payload.pop("baseline_id")
    incomplete_path = workdir / "rejected-incomplete-run-manifest-v2.json"
    incomplete_path.write_text(
        json.dumps(incomplete_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rejections = [
        _expect_failure(
            "run-manifest:incomplete",
            lambda: load_run_manifest_v2(incomplete_path),
        ),
        _expect_failure(
            "run-manifest:dirty-primary",
            lambda: _require_promotable(replace(manifest, dirty=True)),
        ),
    ]
    return {
        "manifest_id": restored.manifest_id,
        "evidence_fingerprint": restored.evidence_fingerprint,
        "manifest_path": str(manifest_path),
        "rejections": rejections,
    }
