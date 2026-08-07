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


_STATUS_ID_V1 = "causal4d-project-status-v1"
_STATUS_ID_V2 = "causal4d-project-status-v2"
_EXPECTED_CLAIM_STATUS = "controlled_passed_real_pending"
_EXPECTED_NEXT_MILESTONE = "same_object_multi_action_real_protocol"
_EXPECTED_EMPIRICAL_STATUS_V1 = {
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
_EXPECTED_NON_CLAIMS_TAIL_V2 = (
    "independent-execution calibration is not yet established",
    "the controlled synthetic Prob4D-to-Bayesian-PhysTwin result does not "
    "establish fresh held-out physical-prediction benefit",
    "a fresh real Prob4D provider is not admitted into the confirmatory "
    "physical estimator without its source-calibrated gate",
    "semantic reweighting is not admitted into the primary method",
)
_EXPECTED_CAUSAL4D_CONTROLLED = {
    "evidence_repository": "IPS-Stuttgart/Causal4D",
    "evidence_revision": "3e63976e01f2ee0624606a85f814002d48a2ad59",
    "milestone": "v0.3.0-causal4d-aip",
    "registered_gate_count": 13,
    "scope": "controlled_counterfactual",
    "status": "passed",
}
_EXPECTED_PROB4D_CONTROLLED = {
    "bayesian_phystwin_execution_revision": (
        "04cc243aea82bfec1b8a2481ef99b38b357e4123"
    ),
    "evidence_repository": "IPS-Stuttgart/BayesianPhysTwin",
    "evidence_revision": "db0f0119a3a4220f5489566829846681e844627d",
    "prob4d_revision": "aa8ffc6541011d044561e09870569a14ab3f586f",
    "protocol_sha256": (
        "921da8a6f14f9430b3f4861d68326d904f61b922e3aedd2b35882ea97bc63111"
    ),
    "report_id": ("c592807d62e9f5121acf85747432574601264160de67b15e9a1c8e48a12cc040"),
    "status": "passed",
}
_TOP_LEVEL_FIELDS_V1 = frozenset(
    {
        "claim_status",
        "empirical_status",
        "non_claims",
        "packages",
        "primary_next_milestone",
        "schema_version",
        "snapshot_date",
        "status_id",
    }
)
_TOP_LEVEL_FIELDS_V2 = frozenset(
    {
        "claim_status",
        "empirical_status",
        "non_claims",
        "packages",
        "primary_next_milestone",
        "schema_version",
        "snapshot_date",
        "status_id",
        "supersedes_status_id",
    }
)
_EMPIRICAL_FIELDS_V2 = frozenset(
    {
        "causal4d_confirmatory_physical",
        "causal4d_controlled",
        "independent_execution_calibration",
        "prob4d_to_bayesian_phystwin",
        "semantic_reweighting",
    }
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _closed_mapping(
    value: Any,
    *,
    name: str,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    record = _mapping(value, name=name)
    missing = sorted(fields - set(record))
    extra = sorted(set(record) - fields)
    require(
        not missing and not extra,
        f"{name} fields changed; missing={missing}, extra={extra}",
    )
    return record


def _nonempty_text(value: Any, *, name: str) -> str:
    require(type(value) is str and bool(value.strip()), f"{name} must be nonempty")
    return value.strip()


def _specifier(value: Any, *, name: str) -> str:
    text = _nonempty_text(value, name=name)
    try:
        SpecifierSet(text)
    except InvalidSpecifier as error:
        raise RuntimeError(
            f"{name} is not a valid version specifier: {text}"
        ) from error
    return text


def _version(value: Any, *, name: str) -> str:
    text = _nonempty_text(value, name=name)
    try:
        Version(text)
    except InvalidVersion as error:
        raise RuntimeError(f"{name} is not a valid version: {text}") from error
    return text


def _iso_date(value: Any, *, name: str) -> date:
    text = _nonempty_text(value, name=name)
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise RuntimeError(f"{name} must use ISO YYYY-MM-DD format") from error


def _integer_count(value: Any, *, name: str) -> int:
    require(
        type(value) is int and value >= 0,
        f"{name} must be a nonnegative integer",
    )
    return value


def _optional_sha256(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be null or a lowercase SHA-256 digest",
    )
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    require(
        path.is_file() and not path.is_symlink(),
        f"project status does not exist or is not a regular file: {path}",
    )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"project status contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RuntimeError(f"project status contains non-finite value: {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise RuntimeError("project status must be UTF-8 JSON") from error
    return dict(_mapping(payload, name="project status"))


def _validate_packages(status: Mapping[str, Any]) -> Mapping[str, Any]:
    packages = _closed_mapping(
        status.get("packages"),
        name="packages",
        fields=frozenset(_EXPECTED_PACKAGE_ROLES),
    )
    for package_name, expected_role in _EXPECTED_PACKAGE_ROLES.items():
        expected_fields = (
            frozenset({"required_version", "role"})
            if package_name == "causal4d"
            else frozenset({"role", "supported_versions"})
        )
        record = _closed_mapping(
            packages[package_name],
            name=f"packages.{package_name}",
            fields=expected_fields,
        )
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
    return packages


def _validate_common(
    status: Mapping[str, Any],
    *,
    status_id: str,
) -> None:
    require(status.get("status_id") == status_id, "unexpected status_id")
    require(
        status.get("claim_status") == _EXPECTED_CLAIM_STATUS,
        "project status overstates or changes the registered claim boundary",
    )
    require(
        status.get("primary_next_milestone") == _EXPECTED_NEXT_MILESTONE,
        "the decisive same-object real protocol is no longer the next milestone",
    )
    _iso_date(status.get("snapshot_date"), name="snapshot_date")
    _validate_packages(status)
    non_claims = status.get("non_claims")
    require(
        isinstance(non_claims, list) and len(non_claims) >= 4,
        "project status must retain its explicit non-claims",
    )
    require(
        all(type(value) is str and value.strip() for value in non_claims),
        "project-status non-claims must be nonempty strings",
    )


def _validate_v1(status: dict[str, Any]) -> dict[str, Any]:
    _closed_mapping(status, name="project status", fields=_TOP_LEVEL_FIELDS_V1)
    require(status.get("schema_version") == 1, "unsupported project-status schema")
    _validate_common(status, status_id=_STATUS_ID_V1)
    empirical = dict(_mapping(status.get("empirical_status"), name="empirical_status"))
    require(
        empirical == _EXPECTED_EMPIRICAL_STATUS_V1,
        "empirical status changed without updating the versioned status contract",
    )
    return status


def _validate_v2(status: dict[str, Any]) -> dict[str, Any]:
    _closed_mapping(status, name="project status", fields=_TOP_LEVEL_FIELDS_V2)
    require(status.get("schema_version") == 2, "unsupported project-status schema")
    require(
        status.get("supersedes_status_id") == _STATUS_ID_V1,
        "project-status v2 must explicitly supersede v1",
    )
    _validate_common(status, status_id=_STATUS_ID_V2)
    empirical = _closed_mapping(
        status.get("empirical_status"),
        name="empirical_status",
        fields=_EMPIRICAL_FIELDS_V2,
    )
    controlled = _closed_mapping(
        empirical["causal4d_controlled"],
        name="empirical_status.causal4d_controlled",
        fields=frozenset(_EXPECTED_CAUSAL4D_CONTROLLED),
    )
    require(
        dict(controlled) == _EXPECTED_CAUSAL4D_CONTROLLED,
        "controlled Causal4D evidence binding changed",
    )

    physical = _closed_mapping(
        empirical["causal4d_confirmatory_physical"],
        name="empirical_status.causal4d_confirmatory_physical",
        fields=frozenset(
            {
                "acquired_executions",
                "claim_ready",
                "evidence_status_sha256",
                "specified_executions",
                "status",
                "validated_executions",
            }
        ),
    )
    require(
        physical.get("status") == "pending",
        "project-status v2 is pre-acquisition; a completed result needs a new schema",
    )
    specified = _integer_count(
        physical.get("specified_executions"),
        name="causal4d_confirmatory_physical.specified_executions",
    )
    acquired = _integer_count(
        physical.get("acquired_executions"),
        name="causal4d_confirmatory_physical.acquired_executions",
    )
    validated = _integer_count(
        physical.get("validated_executions"),
        name="causal4d_confirmatory_physical.validated_executions",
    )
    require(
        specified == 36,
        "the registered physical protocol must specify 36 executions",
    )
    require(
        validated <= acquired <= specified,
        "physical execution accounting must satisfy validated <= acquired <= specified",
    )
    require(
        physical.get("claim_ready") is False,
        "project-status v2 cannot mark the confirmatory result claim-ready",
    )
    evidence_status_sha256 = _optional_sha256(
        physical.get("evidence_status_sha256"),
        name="causal4d_confirmatory_physical.evidence_status_sha256",
    )
    if acquired == 0:
        require(
            validated == 0 and evidence_status_sha256 is None,
            "zero acquired executions cannot carry validated evidence or a digest",
        )
    else:
        require(
            evidence_status_sha256 is not None,
            "nonzero physical progress requires a bound evidence-status digest",
        )
    expected_non_claims = (
        "the registered same-object physical experiment remains at "
        f"{acquired}/{specified} acquired and {validated}/{specified} "
        "validated executions",
        *_EXPECTED_NON_CLAIMS_TAIL_V2,
    )
    require(
        tuple(status["non_claims"]) == expected_non_claims,
        "project-status v2 non-claims do not match its evidence boundary",
    )

    calibration = _closed_mapping(
        empirical["independent_execution_calibration"],
        name="empirical_status.independent_execution_calibration",
        fields=frozenset({"evidence_sha256", "status"}),
    )
    require(
        calibration.get("status") == "pending"
        and calibration.get("evidence_sha256") is None,
        "project-status v2 keeps independent-execution calibration pending",
    )

    prob4d = _closed_mapping(
        empirical["prob4d_to_bayesian_phystwin"],
        name="empirical_status.prob4d_to_bayesian_phystwin",
        fields=frozenset(
            {
                "confirmatory_physical_use",
                "controlled_synthetic",
                "fresh_real_provider",
            }
        ),
    )
    controlled_prob4d = _closed_mapping(
        prob4d["controlled_synthetic"],
        name="prob4d_to_bayesian_phystwin.controlled_synthetic",
        fields=frozenset(_EXPECTED_PROB4D_CONTROLLED),
    )
    require(
        dict(controlled_prob4d) == _EXPECTED_PROB4D_CONTROLLED,
        "controlled synthetic Prob4D evidence binding changed",
    )
    fresh_prob4d = _closed_mapping(
        prob4d["fresh_real_provider"],
        name="prob4d_to_bayesian_phystwin.fresh_real_provider",
        fields=frozenset({"evidence_sha256", "status"}),
    )
    require(
        fresh_prob4d.get("status") == "pending"
        and fresh_prob4d.get("evidence_sha256") is None,
        "fresh real Prob4D-provider evidence changed without a new status schema",
    )
    require(
        prob4d.get("confirmatory_physical_use") == "not_admitted",
        "Prob4D confirmatory use was admitted without its fresh real gate",
    )

    semantic = _closed_mapping(
        empirical["semantic_reweighting"],
        name="empirical_status.semantic_reweighting",
        fields=frozenset({"status"}),
    )
    require(
        semantic.get("status") == "not_admitted",
        "semantic reweighting was admitted into the primary method",
    )
    return status


def load_project_status(path: Path) -> dict[str, Any]:
    """Load and fail closed on unsupported or claim-inflating status records."""

    status = _load_json_object(path)
    schema_version = status.get("schema_version")
    if schema_version == 1:
        return _validate_v1(status)
    if schema_version == 2:
        return _validate_v2(status)
    raise RuntimeError("unsupported project-status schema")


def validate_project_status_transition(
    previous_path: Path,
    current_path: Path,
) -> dict[str, Any]:
    """Validate one explicit, monotone project-status transition."""

    previous = load_project_status(previous_path)
    current = load_project_status(current_path)
    require(
        _iso_date(current["snapshot_date"], name="current.snapshot_date")
        >= _iso_date(previous["snapshot_date"], name="previous.snapshot_date"),
        "project-status snapshot date moved backwards",
    )
    previous_packages = _mapping(previous["packages"], name="previous.packages")
    current_packages = _mapping(current["packages"], name="current.packages")
    for package_name, expected_role in _EXPECTED_PACKAGE_ROLES.items():
        require(
            _mapping(
                previous_packages[package_name],
                name=f"previous.packages.{package_name}",
            ).get("role")
            == expected_role
            == _mapping(
                current_packages[package_name],
                name=f"current.packages.{package_name}",
            ).get("role"),
            f"packages.{package_name}.role changed across status transition",
        )

    if previous["schema_version"] == 1 and current["schema_version"] == 2:
        require(
            current["supersedes_status_id"] == previous["status_id"],
            "project-status v2 does not supersede the supplied v1 record",
        )
        previous_empirical = _mapping(
            previous["empirical_status"],
            name="previous.empirical_status",
        )
        current_empirical = _mapping(
            current["empirical_status"],
            name="current.empirical_status",
        )
        require(
            _mapping(
                current_empirical["causal4d_controlled"],
                name="current causal4d controlled",
            )["status"]
            == previous_empirical["controlled_counterfactual"],
            "controlled Causal4D status changed across the v1-to-v2 split",
        )
        require(
            _mapping(
                current_empirical["causal4d_confirmatory_physical"],
                name="current physical status",
            )["status"]
            == previous_empirical["same_object_multi_action_real"],
            "physical status changed across the v1-to-v2 split",
        )
        require(
            _mapping(
                current_empirical["independent_execution_calibration"],
                name="current calibration status",
            )["status"]
            == previous_empirical["independent_execution_calibration"],
            "calibration status changed across the v1-to-v2 split",
        )
        require(
            _mapping(
                current_empirical["semantic_reweighting"],
                name="current semantic status",
            )["status"]
            == previous_empirical["semantic_reweighting"],
            "semantic status changed across the v1-to-v2 split",
        )
        require(
            previous_empirical["prob4d_to_bayesian_phystwin"] == "prospective_pending"
            and _mapping(
                _mapping(
                    current_empirical["prob4d_to_bayesian_phystwin"],
                    name="current Prob4D status",
                )["fresh_real_provider"],
                name="current fresh-real Prob4D status",
            )["status"]
            == "pending",
            "v2 must retain the pending fresh-real Prob4D evidence boundary",
        )
        return {
            "from_status_id": previous["status_id"],
            "to_status_id": current["status_id"],
            "transition": "v1_to_v2_evidence_split",
        }

    require(
        previous["schema_version"] == current["schema_version"] == 2,
        "unsupported project-status transition",
    )
    previous_empirical = _mapping(
        previous["empirical_status"],
        name="previous empirical",
    )
    current_empirical = _mapping(
        current["empirical_status"],
        name="current empirical",
    )
    previous_physical = _mapping(
        previous_empirical["causal4d_confirmatory_physical"],
        name="previous physical",
    )
    current_physical = _mapping(
        current_empirical["causal4d_confirmatory_physical"],
        name="current physical",
    )
    require(
        current_physical["specified_executions"]
        == previous_physical["specified_executions"],
        "physical specified_executions changed across status transition",
    )
    for field in ("acquired_executions", "validated_executions"):
        require(
            current_physical[field] >= previous_physical[field],
            f"physical {field} moved backwards",
        )
    return {
        "from_status_id": previous["status_id"],
        "to_status_id": current["status_id"],
        "transition": "monotone_v2_progress_update",
    }


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
    summary: dict[str, Any] = {
        "claim_status": status["claim_status"],
        "empirical_status": status["empirical_status"],
        "primary_next_milestone": status["primary_next_milestone"],
        "schema_version": status["schema_version"],
        "status_id": status["status_id"],
        "status_sha256": _canonical_sha256(status),
        "versions": {"causal4d": installed_causal4d},
    }
    if status["schema_version"] == 2:
        physical = _mapping(
            _mapping(status["empirical_status"], name="empirical_status")[
                "causal4d_confirmatory_physical"
            ],
            name="causal4d_confirmatory_physical",
        )
        summary["physical_confirmatory"] = {
            "acquired_executions": physical["acquired_executions"],
            "claim_ready": physical["claim_ready"],
            "specified_executions": physical["specified_executions"],
            "validated_executions": physical["validated_executions"],
        }
    return summary


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
    parser.add_argument("--previous-status", type=Path)
    parser.add_argument("--installed-stack", action="store_true")
    arguments = parser.parse_args()
    if arguments.previous_status is not None:
        print(
            json.dumps(
                validate_project_status_transition(
                    arguments.previous_status.resolve(),
                    arguments.status.resolve(),
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    validator = (
        validate_installed_stack_status
        if arguments.installed_stack
        else validate_causal4d_status
    )
    print(json.dumps(validator(arguments.status.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
