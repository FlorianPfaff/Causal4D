"""Installed-wheel golden path for Prob4D, Bayesian-PhysTwin, and Causal4D.

The runner is copied outside every source checkout before execution. It accepts
only data and exact repository revisions from those checkouts; all Python imports
must resolve from a clean virtual environment containing built wheels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from three_repository_common import installed_package_origins, require
from three_repository_manifest import run_evidence_manifest_checks
from three_repository_observation import (
    fixture_artifact,
    roundtrip_prob4d_artifact,
    run_bpt_update,
    run_rejection_corpus,
)
from three_repository_rollout import run_causal4d_rollout
from three_repository_status import validate_installed_stack_status


def _require_runner_outside_checkouts(checkout_roots: tuple[Path, ...]) -> None:
    runner = Path(__file__).resolve()
    for root in checkout_roots:
        try:
            runner.relative_to(root.resolve())
        except ValueError:
            continue
        raise RuntimeError(f"golden-path runner is inside a checkout: {runner}")


def _default_project_status() -> str:
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return str(Path(workspace) / "causal4d" / "ci" / "project_status_v2.json")
    return "ci/project_status_v2.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixture_path = Path(args.prob4d_fixture).resolve()
    project_status_path = Path(args.project_status).resolve()
    output_path = Path(args.output).resolve()
    checkout_roots = tuple(Path(value).resolve() for value in args.checkout_root)
    require(fixture_path.is_file(), f"fixture does not exist: {fixture_path}")
    require(
        project_status_path.is_file(),
        f"project status does not exist: {project_status_path}",
    )
    require(checkout_roots, "at least one checkout root must be supplied")
    require(os.environ.get("PYTHONPATH") in {None, ""}, "PYTHONPATH must be unset")
    _require_runner_outside_checkouts(checkout_roots)
    origins = installed_package_origins(checkout_roots)
    project_status = validate_installed_stack_status(project_status_path)

    workdir = output_path.parent / "three-repository-golden-work"
    workdir.mkdir(parents=True, exist_ok=True)
    artifact, fixture_payload = fixture_artifact(fixture_path)
    observation_path = workdir / "observation-belief.npz"
    restored_prob4d, prob4d_manifest = roundtrip_prob4d_artifact(
        artifact,
        observation_path,
    )
    require(
        restored_prob4d.artifact_id == fixture_payload["expected_artifact_id"],
        "producer round trip changed the fixture ID",
    )

    bpt_result, bpt_summary = run_bpt_update(observation_path)
    rejection_corpus = run_rejection_corpus(
        artifact,
        observation_path,
        workdir,
    )
    causal4d_summary = run_causal4d_rollout(
        observation_path,
        bpt_result,
        bpt_summary,
        workdir,
    )
    evidence_manifest = run_evidence_manifest_checks(
        workdir=workdir,
        observation_path=observation_path,
        twin_belief_path=Path(causal4d_summary["twin_belief_path"]),
        rollout_bank_path=Path(causal4d_summary["rollout_bank_path"]),
        revisions={
            "prob4d": args.prob4d_revision,
            "bayesian-phystwin": args.bpt_revision,
            "causal4d": args.causal4d_revision,
        },
        provider_manifest_id=causal4d_summary["provider_manifest_id"],
        observation_artifact_id=restored_prob4d.artifact_id,
    )
    summary = {
        "status": "passed",
        "schema_version": 1,
        "package_origins": origins,
        "project_status": project_status,
        "repository_revisions": {
            "FlorianPfaff/Prob4D": args.prob4d_revision,
            "FlorianPfaff/Bayesian-PhysTwin": args.bpt_revision,
            "FlorianPfaff/Causal4D": args.causal4d_revision,
        },
        "observation_artifact_id": restored_prob4d.artifact_id,
        "prob4d_provider_manifest": prob4d_manifest,
        "bpt_update": bpt_summary,
        "causal4d": causal4d_summary,
        "evidence_manifest": evidence_manifest,
        "rejection_corpus": rejection_corpus,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob4d-fixture", required=True)
    parser.add_argument("--project-status", default=_default_project_status())
    parser.add_argument("--prob4d-revision", required=True)
    parser.add_argument("--bpt-revision", required=True)
    parser.add_argument("--causal4d-revision", required=True)
    parser.add_argument(
        "--checkout-root",
        action="append",
        default=[],
        help="Source checkout root that installed imports must not resolve under.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
