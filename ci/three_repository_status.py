"""Validate the machine-readable Causal4D cross-repository project status."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from three_repository_common import require


_EXPECTED_STATUS_ID = "causal4d-project-status-v1"
_EXPECTED_CLAIM_STATUS = "controlled_passed_real_pending"
_EXPECTED_NEXT_MILESTONE = "same_object_multi_action_real_protocol"
_EXPECTED_EMPIRICAL_STATUS = {
    "controlled_counterfactual": "passed",
    "independent_execution_calibration": "pending",
    "prob4d_to_bayesian_phystwin": "prospective_pending",
    "same_object_multi_action_real": "pending",
    "semantic_reweighting": "not_admitted",
}
_EXPECTED_PACKAGE_ROLES = {
    "bayesian-phystwin": "uncertain_physical_twin",
    "causal4d": "realized_intervention_abduction_and_prediction",
    "prob4d": "optional_probabilistic_observation_feeder",
}


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _nonempty_text(value: Any, *, name: str) -> str:
    text = str(value).strip()
    require(bool(text), f"{name} must be nonempty")
    return text


def _specifier(value: Any, *, name: str) -> str:
    text = _nonempty_text(value, name=name)
    try:
        SpecifierSet(text)
    except InvalidSpecifier as error:
        raise RuntimeError(f"{name} is not a valid version specifier: {text}") from error
    return text


def _version(value: Any, *, name: str) -> str:
    text = _nonempty_text(value, name=name)
    try:
        Version(text)
    except InvalidVersion as error:
        raise RuntimeError(f"{name} is not a valid version: {text}") from error
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_project_status(path: Path) -> dict[str, Any]:
    """Load and fail closed on unsupported or claim-inflating status records."""

    require(path.is_file(), f"project status does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = dict(_mapping(payload, name="project status"))
    require(status.get("schema_version") == 1, "unsupported project-status schema")
    require(status.get("status_id") == _EXPECTED_STATUS_ID, "unexpected status_id")
    require(
        status.get("claim_status") == _EXPECTED_CLAIM_STATUS,
        "project status overstates or changes the registered claim boundary",
    )
    require(
        status.get("primary_next_milestone") == _EXPECTED_NEXT_MILESTONE,
        "the decisive same-object real protocol is no longer the next milestone",
    )
    try:
        date.fromisoformat(_nonempty_text(status.get("snapshot_date"), name="snapshot_date"))
    except ValueError as error:
        raise RuntimeError("snapshot_date must use ISO YYYY-MM-DD format") from error

    empirical = dict(
        _mapping(status.get("empirical_status"), name="empirical_status")
    )
    require(
        empirical == _EXPECTED_EMPIRICAL_STATUS,
        "empirical status changed without updating the versioned status contract",
    )

    packages = dict(_mapping(status.get("packages"), name="packages"))
    require(
        set(packages) == set(_EXPECTED_PACKAGE_ROLES),
        "project status must describe exactly Causal4D, Bayesian-PhysTwin, and Prob4D",
    )
    for package_name, expected_role in _EXPECTED_PACKAGE_ROLES.items():
        record = _mapping(packages[package_name], name=f"packages.{package_name}")
        require(
            record.get("role") == expected_role,
            f"packages.{package_name}.role changed",
        )
    causal4d_record = _mapping(packages["causal4d"], name="packages.causal4d")
    _version(
        causal4d_record.get("required_version"),
        name="packages.causal4d.required_version",
    )
    for package_name in ("bayesian-phystwin", "prob4d"):
        record = _mapping(packages[package_name], name=f"packages.{package_name}")
        _specifier(
            record.get("supported_versions"),
            name=f"packages.{package_name}.supported_versions",
        )

    non_claims = status.get("non_claims")
    require(
        isinstance(non_claims, list) and len(non_claims) >= 4,
        "project status must retain its explicit non-claims",
    )
    require(
        all(isinstance(value, str) and value.strip() for value in non_claims),
        "project-status non-claims must be nonempty strings",
    )
    return status


def validate_causal4d_status(path: Path) -> dict[str, Any]:
    """Validate Causal4D's installed version and provider range against status."""

    import causal4d
    from causal4d.provider_contract import BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE

    status = load_project_status(path)
    packages = _mapping(status["packages"], name="packages")
    causal4d_record = _mapping(packages["causal4d"], name="packages.causal4d")
    required_causal4d = _version(
        causal4d_record["required_version"],
        name="packages.causal4d.required_version",
    )
    installed_causal4d = importlib.metadata.version("causal4d")
    require(
        installed_causal4d == required_causal4d,
        "installed Causal4D version differs from the project-status contract",
    )
    require(
        causal4d.__version__ == required_causal4d,
        "causal4d.__version__ differs from the project-status contract",
    )

    bpt_record = _mapping(
        packages["bayesian-phystwin"],
        name="packages.bayesian-phystwin",
    )
    supported_bpt = _specifier(
        bpt_record["supported_versions"],
        name="packages.bayesian-phystwin.supported_versions",
    )
    require(
        supported_bpt == BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        "Bayesian-PhysTwin compatibility range drifted from provider_contract.py",
    )
    return {
        "claim_status": status["claim_status"],
        "empirical_status": status["empirical_status"],
        "primary_next_milestone": status["primary_next_milestone"],
        "status_id": status["status_id"],
        "status_sha256": _canonical_sha256(status),
        "versions": {"causal4d": installed_causal4d},
    }


def validate_installed_stack_status(path: Path) -> dict[str, Any]:
    """Validate all three installed wheels against the shared status contract."""

    summary = validate_causal4d_status(path)
    status = load_project_status(path)
    packages = _mapping(status["packages"], name="packages")
    versions = {
        package_name: importlib.metadata.version(package_name)
        for package_name in ("bayesian-phystwin", "causal4d", "prob4d")
    }
    for package_name in ("bayesian-phystwin", "prob4d"):
        record = _mapping(packages[package_name], name=f"packages.{package_name}")
        supported = _specifier(
            record["supported_versions"],
            name=f"packages.{package_name}.supported_versions",
        )
        require(
            Version(versions[package_name]) in SpecifierSet(supported),
            f"installed {package_name} {versions[package_name]} is outside {supported}",
        )
    summary["versions"] = versions
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--installed-stack", action="store_true")
    arguments = parser.parse_args()
    validator = (
        validate_installed_stack_status
        if arguments.installed_stack
        else validate_causal4d_status
    )
    print(json.dumps(validator(arguments.status.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
