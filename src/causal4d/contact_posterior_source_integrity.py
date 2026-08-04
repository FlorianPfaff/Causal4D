"""Fail-closed source-bundle validation for contact-posterior diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


_BENCHMARK = "causal4d-latent-contact-v1"
_EXPECTED_ARTIFACTS = frozenset(
    {
        "summary.json",
        "protocol.json",
        "interventions.csv",
        "contact_recovery.csv",
        "fold_calibration.csv",
        "success_gates.json",
    }
)
_RECOVERY_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "object",
        "source_objects",
        "world_condition",
        "setting",
        "observation_fraction",
        "node_truth",
        "node_map",
        "node_correct",
        "node_confidence",
        "node_truth_probability",
        "node_brier",
        "node_credible_covered",
        "delay_map",
        "delay_map_correct",
        "joint_effective_sample_size",
        "joint_normalized_entropy",
    }
)
_INTERVENTION_REQUIRED_FIELDS = frozenset(
    {
        "seed",
        "object",
        "source_objects",
        "world_condition",
        "setting",
        "method",
        "observation_fraction",
        "trajectory_rmse_m",
    }
)
_SETTINGS = frozenset({"pre_intervention", "online_adaptation"})
_WORLD_CONDITIONS = frozenset({"matched_contact", "shifted_contact"})
_REQUIRED_TRAJECTORY_METHODS = frozenset({"nominal_physics", "latent_contact"})

_RowKey = tuple[int, str, str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _require_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"source bundle artifact must not be a symlink: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"source bundle artifact is missing: {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    _require_regular_file(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _safe_artifact_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("manifest artifact names must be strings")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name in {"", ".", ".."}
        or "\\" in value
    ):
        raise ValueError(f"unsafe manifest artifact name: {value!r}")
    return path.name


def _read_csv_rows(
    path: Path,
) -> tuple[tuple[str, ...], list[tuple[int, dict[str, str]]]]:
    _require_regular_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV has no header: {path.name}") from error
        header = tuple(raw_header)
        if not header or any(not field for field in header):
            raise ValueError(f"CSV header contains an empty field: {path.name}")
        if len(set(header)) != len(header):
            raise ValueError(f"CSV header contains duplicate fields: {path.name}")
        rows: list[tuple[int, dict[str, str]]] = []
        for line_number, values in enumerate(reader, start=2):
            if not values or all(value == "" for value in values):
                raise ValueError(f"blank CSV row at {path.name}:{line_number}")
            if len(values) != len(header):
                raise ValueError(
                    f"CSV row width differs from its header at "
                    f"{path.name}:{line_number}"
                )
            rows.append(
                (
                    line_number,
                    dict(zip(header, values, strict=True)),
                )
            )
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path.name}")
    return header, rows


def _require_headers(
    path: Path,
    header: tuple[str, ...],
    required: frozenset[str],
) -> None:
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"{path.name} is missing required fields: {missing!r}")


def _parse_int(value: str, *, field: str, location: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{location}/{field} must be an integer") from error
    if value != str(parsed):
        raise ValueError(f"{location}/{field} must use canonical integer syntax")
    return parsed


def _parse_finite_float(value: str, *, field: str, location: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{location}/{field} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{location}/{field} must be finite")
    return parsed


def _parse_bool(value: str, *, field: str, location: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{location}/{field} must be serialized as True or False")


def _parse_nodes(value: str, *, field: str, location: str) -> tuple[int, ...]:
    if not value:
        raise ValueError(f"{location}/{field} must be nonempty")
    nodes = tuple(
        _parse_int(part, field=field, location=location) for part in value.split(";")
    )
    if any(node < 0 for node in nodes) or len(set(nodes)) != len(nodes):
        raise ValueError(f"{location}/{field} contains invalid contact nodes")
    return nodes


def _parse_sources(value: str, *, object_name: str, location: str) -> tuple[str, ...]:
    sources = tuple(value.split(";"))
    if not sources or any(not source for source in sources):
        raise ValueError(f"{location}/source_objects must be nonempty")
    if len(set(sources)) != len(sources):
        raise ValueError(f"{location}/source_objects contains duplicates")
    if object_name in sources:
        raise ValueError(f"{location}/source_objects contains the held-out object")
    return sources


def _row_key(row: Mapping[str, str], *, location: str) -> _RowKey:
    seed = _parse_int(row["seed"], field="seed", location=location)
    object_name = row["object"]
    world = row["world_condition"]
    setting = row["setting"]
    if not object_name:
        raise ValueError(f"{location}/object must be nonempty")
    if world not in _WORLD_CONDITIONS:
        raise ValueError(f"{location}/world_condition is unsupported: {world!r}")
    if setting not in _SETTINGS:
        raise ValueError(f"{location}/setting is unsupported: {setting!r}")
    return seed, object_name, world, setting


def _validate_observation_fraction(
    value: str,
    *,
    setting: str,
    expected_online: float,
    location: str,
) -> None:
    observed = _parse_finite_float(
        value,
        field="observation_fraction",
        location=location,
    )
    expected = expected_online if setting == "online_adaptation" else 0.0
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"{location}/observation_fraction differs from the summary: "
            f"{observed!r} != {expected!r}"
        )


def _validate_manifest(bundle: Path) -> tuple[dict[str, Any], str]:
    manifest_path = bundle / "manifest.json"
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("result manifest schema_version must equal 1")
    if manifest.get("benchmark") != _BENCHMARK:
        raise ValueError("result manifest benchmark is unsupported")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise ValueError("result manifest artifacts must be a nonempty object")

    artifacts: dict[str, Mapping[str, Any]] = {}
    for raw_name, raw_record in raw_artifacts.items():
        name = _safe_artifact_name(raw_name)
        if name in artifacts:
            raise ValueError(f"duplicate normalized manifest artifact name: {name!r}")
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"artifact record for {name!r} must be an object")
        artifacts[name] = raw_record
    if set(artifacts) != _EXPECTED_ARTIFACTS:
        missing = sorted(_EXPECTED_ARTIFACTS - set(artifacts))
        extra = sorted(set(artifacts) - _EXPECTED_ARTIFACTS)
        raise ValueError(
            "result manifest payload inventory differs from the locked schema; "
            f"missing={missing!r}, extra={extra!r}"
        )

    expected_entries = set(_EXPECTED_ARTIFACTS) | {"manifest.json"}
    actual_entries = {entry.name for entry in bundle.iterdir()}
    if actual_entries != expected_entries:
        missing = sorted(expected_entries - actual_entries)
        extra = sorted(actual_entries - expected_entries)
        raise ValueError(
            "source bundle file inventory differs from its locked schema; "
            f"missing={missing!r}, extra={extra!r}"
        )

    for name, record in artifacts.items():
        expected_hash = record.get("sha256")
        expected_bytes = record.get("bytes")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"artifact {name!r} has an invalid SHA-256 digest")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError(f"artifact {name!r} has an invalid byte count")
        path = bundle / name
        _require_regular_file(path)
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"artifact {name!r} byte count changed: "
                f"{actual_bytes} != {expected_bytes}"
            )
        if actual_hash != expected_hash:
            raise ValueError(
                f"artifact {name!r} checksum changed: {actual_hash} != {expected_hash}"
            )
    return manifest, _sha256(manifest_path)


def _validate_summary(bundle: Path) -> tuple[dict[str, Any], tuple[int, ...], float]:
    summary = _read_json_object(bundle / "summary.json")
    if summary.get("schema_version") != 1:
        raise ValueError("summary schema_version must equal 1")
    if summary.get("benchmark") != _BENCHMARK:
        raise ValueError("summary benchmark is unsupported")
    raw_seeds = summary.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("summary seeds must be a nonempty list")
    seeds: list[int] = []
    for value in raw_seeds:
        if type(value) is not int or value < 0:
            raise ValueError("summary seeds must be nonnegative integers")
        seeds.append(value)
    if len(set(seeds)) != len(seeds):
        raise ValueError("summary seeds must be unique")

    for field in ("benchmark_config", "contact_config", "success_gates"):
        if not isinstance(summary.get(field), Mapping):
            raise ValueError(f"summary {field} must be an object")
    contact_config = summary["contact_config"]
    raw_fraction = contact_config.get("observation_fraction")
    if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, (int, float)):
        raise ValueError("summary observation_fraction must be numeric")
    observation_fraction = float(raw_fraction)
    if not math.isfinite(observation_fraction) or not 0.0 < observation_fraction < 1.0:
        raise ValueError("summary observation_fraction must be finite and in (0, 1)")

    success_gates = _read_json_object(bundle / "success_gates.json")
    if success_gates != summary["success_gates"]:
        raise ValueError("summary and success_gates.json disagree")
    return summary, tuple(seeds), observation_fraction


def _validate_recovery_rows(
    bundle: Path,
    *,
    seeds: tuple[int, ...],
    observation_fraction: float,
) -> tuple[dict[_RowKey, tuple[str, ...]], int]:
    path = bundle / "contact_recovery.csv"
    header, rows = _read_csv_rows(path)
    _require_headers(path, header, _RECOVERY_REQUIRED_FIELDS)
    seed_set = set(seeds)
    seen: set[_RowKey] = set()
    online: dict[_RowKey, tuple[str, ...]] = {}
    observed_seeds: set[int] = set()
    for line_number, row in rows:
        location = f"{path.name}:{line_number}"
        key = _row_key(row, location=location)
        if key in seen:
            raise ValueError(f"duplicate contact-recovery row key: {key!r}")
        seen.add(key)
        seed, object_name, _, setting = key
        if seed not in seed_set:
            raise ValueError(f"{location}/seed is not declared by summary.json")
        observed_seeds.add(seed)
        sources = _parse_sources(
            row["source_objects"],
            object_name=object_name,
            location=location,
        )
        _validate_observation_fraction(
            row["observation_fraction"],
            setting=setting,
            expected_online=observation_fraction,
            location=location,
        )
        _parse_nodes(row["node_truth"], field="node_truth", location=location)
        _parse_nodes(row["node_map"], field="node_map", location=location)
        _parse_bool(row["node_correct"], field="node_correct", location=location)
        _parse_bool(
            row["node_credible_covered"],
            field="node_credible_covered",
            location=location,
        )
        _parse_bool(
            row["delay_map_correct"],
            field="delay_map_correct",
            location=location,
        )
        _parse_int(row["delay_map"], field="delay_map", location=location)
        for field in (
            "node_confidence",
            "node_truth_probability",
            "node_brier",
            "joint_effective_sample_size",
            "joint_normalized_entropy",
        ):
            value = _parse_finite_float(row[field], field=field, location=location)
            if field in {"node_confidence", "node_truth_probability"} and not (
                0.0 <= value <= 1.0
            ):
                raise ValueError(f"{location}/{field} must be in [0, 1]")
            if field == "node_brier" and value < 0.0:
                raise ValueError(f"{location}/{field} must be nonnegative")
            if field == "joint_effective_sample_size" and value <= 0.0:
                raise ValueError(f"{location}/{field} must be positive")
            if field == "joint_normalized_entropy" and not (
                0.0 <= value <= 1.0 + 1e-12
            ):
                raise ValueError(f"{location}/{field} must be in [0, 1]")
        if setting == "online_adaptation":
            online[key] = sources
    if observed_seeds != seed_set:
        raise ValueError("contact_recovery.csv does not cover every declared seed")
    if not online:
        raise ValueError("contact_recovery.csv has no online-adaptation rows")
    return online, len(rows)


def _validate_intervention_rows(
    bundle: Path,
    *,
    seeds: tuple[int, ...],
    observation_fraction: float,
    online_recovery: Mapping[_RowKey, tuple[str, ...]],
) -> int:
    path = bundle / "interventions.csv"
    header, rows = _read_csv_rows(path)
    _require_headers(path, header, _INTERVENTION_REQUIRED_FIELDS)
    seed_set = set(seeds)
    seen: set[tuple[_RowKey, str]] = set()
    methods_by_online_key: dict[_RowKey, set[str]] = {}
    sources_by_key: dict[_RowKey, tuple[str, ...]] = {}
    observed_seeds: set[int] = set()
    for line_number, row in rows:
        location = f"{path.name}:{line_number}"
        key = _row_key(row, location=location)
        seed, object_name, _, setting = key
        if seed not in seed_set:
            raise ValueError(f"{location}/seed is not declared by summary.json")
        observed_seeds.add(seed)
        method = row["method"]
        if not method:
            raise ValueError(f"{location}/method must be nonempty")
        indexed_key = (key, method)
        if indexed_key in seen:
            raise ValueError(f"duplicate intervention row key: {indexed_key!r}")
        seen.add(indexed_key)
        sources = _parse_sources(
            row["source_objects"],
            object_name=object_name,
            location=location,
        )
        previous_sources = sources_by_key.setdefault(key, sources)
        if previous_sources != sources:
            raise ValueError(f"{location}/source_objects changes within one case")
        _validate_observation_fraction(
            row["observation_fraction"],
            setting=setting,
            expected_online=observation_fraction,
            location=location,
        )
        trajectory_rmse = _parse_finite_float(
            row["trajectory_rmse_m"],
            field="trajectory_rmse_m",
            location=location,
        )
        if trajectory_rmse < 0.0:
            raise ValueError(f"{location}/trajectory_rmse_m must be nonnegative")
        if setting == "online_adaptation":
            methods_by_online_key.setdefault(key, set()).add(method)

    if observed_seeds != seed_set:
        raise ValueError("interventions.csv does not cover every declared seed")
    if set(methods_by_online_key) != set(online_recovery):
        missing = sorted(set(online_recovery) - set(methods_by_online_key))
        extra = sorted(set(methods_by_online_key) - set(online_recovery))
        raise ValueError(
            "online intervention and recovery case identities differ; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for key, recovery_sources in online_recovery.items():
        if not _REQUIRED_TRAJECTORY_METHODS.issubset(methods_by_online_key[key]):
            missing_methods = sorted(
                _REQUIRED_TRAJECTORY_METHODS - methods_by_online_key[key]
            )
            raise ValueError(
                f"online intervention case {key!r} is missing paired methods: "
                f"{missing_methods!r}"
            )
        if sources_by_key[key] != recovery_sources:
            raise ValueError(
                f"online intervention and recovery source identities differ for {key!r}"
            )
    return len(rows)


def verify_contact_posterior_source_bundle(
    bundle_directory: str | Path,
) -> dict[str, Any]:
    """Verify exact payload integrity and row identities before diagnostics run."""

    supplied = Path(bundle_directory)
    if supplied.is_symlink():
        raise ValueError("source bundle directory must not be a symlink")
    if not supplied.is_dir():
        raise FileNotFoundError(f"source bundle directory is missing: {supplied}")
    bundle = supplied.resolve()
    manifest, manifest_sha256 = _validate_manifest(bundle)
    summary, seeds, observation_fraction = _validate_summary(bundle)
    if manifest["benchmark"] != summary["benchmark"]:
        raise ValueError("manifest and summary benchmark identities differ")
    online_recovery, recovery_row_count = _validate_recovery_rows(
        bundle,
        seeds=seeds,
        observation_fraction=observation_fraction,
    )
    intervention_row_count = _validate_intervention_rows(
        bundle,
        seeds=seeds,
        observation_fraction=observation_fraction,
        online_recovery=online_recovery,
    )
    return {
        "schema_version": 1,
        "benchmark": _BENCHMARK,
        "manifest_sha256": manifest_sha256,
        "artifact_count": len(_EXPECTED_ARTIFACTS),
        "seed_count": len(seeds),
        "contact_recovery_row_count": recovery_row_count,
        "intervention_row_count": intervention_row_count,
        "online_case_count": len(online_recovery),
        "passed": True,
    }
