"""Strict topology-provenance checks for contact-posterior evidence bundles."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn

from causal4d.contact_posterior_source_integrity import (
    verify_contact_posterior_source_bundle,
)


_BENCHMARK = "causal4d-latent-contact-v1"
_RECOVERY_REQUIRED_FIELDS = frozenset(
    {
        "object",
        "source_objects",
        "held_out_topology",
    }
)
_INTERVENTION_REQUIRED_FIELDS = frozenset(
    {
        "object",
        "source_objects",
        "held_out_topology",
    }
)
_FOLD_REQUIRED_FIELDS = frozenset(
    {
        "object",
        "source_objects",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _require_regular_file(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"provenance artifact must not be a symlink: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(f"provenance artifact is missing: {path}")


def _read_json_value(path: Path) -> Any:
    _require_regular_file(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


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


def _parse_bool(value: str, *, field: str, location: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{location}/{field} must be serialized as True or False")


def _parse_sources(
    value: str,
    *,
    object_name: str,
    location: str,
) -> tuple[str, ...]:
    sources = tuple(value.split(";"))
    if not sources or any(not source for source in sources):
        raise ValueError(f"{location}/source_objects must be nonempty")
    if len(set(sources)) != len(sources):
        raise ValueError(f"{location}/source_objects contains duplicates")
    if object_name in sources:
        raise ValueError(f"{location}/source_objects contains the held-out object")
    return sources


def _expected_protocol_sources(
    summary: Mapping[str, Any],
    protocol: Any,
) -> dict[str, tuple[str, ...]]:
    raw_protocol = summary.get("protocol")
    if not isinstance(raw_protocol, list) or not raw_protocol:
        raise ValueError("summary protocol must be a nonempty list")
    if protocol != raw_protocol:
        raise ValueError("summary and protocol.json disagree")

    expected_sources: dict[str, tuple[str, ...]] = {}
    for index, raw_row in enumerate(raw_protocol):
        location = f"summary.json/protocol/{index}"
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{location} must be an object")
        object_name = raw_row.get("object")
        source_value = raw_row.get("source_objects")
        if not isinstance(object_name, str) or not object_name:
            raise ValueError(f"{location}/object must be nonempty")
        if object_name in expected_sources:
            raise ValueError(f"duplicate protocol object: {object_name!r}")
        if not isinstance(source_value, str):
            raise ValueError(f"{location}/source_objects must be a string")
        expected_sources[object_name] = _parse_sources(
            source_value,
            object_name=object_name,
            location=location,
        )

    protocol_objects = set(expected_sources)
    for object_name, sources in expected_sources.items():
        if set(sources) != protocol_objects - {object_name}:
            raise ValueError(
                f"summary protocol sources for {object_name!r} must name every "
                "other held-in topology exactly once"
            )
    return expected_sources


def _validate_topology_rows(
    path: Path,
    *,
    required: frozenset[str],
    expected_sources: Mapping[str, tuple[str, ...]],
    require_held_out: bool,
) -> int:
    header, rows = _read_csv_rows(path)
    _require_headers(path, header, required)
    for line_number, row in rows:
        location = f"{path.name}:{line_number}"
        object_name = row["object"]
        if object_name not in expected_sources:
            raise ValueError(f"{location}/object is not declared by summary protocol")
        sources = _parse_sources(
            row["source_objects"],
            object_name=object_name,
            location=location,
        )
        if sources != expected_sources[object_name]:
            raise ValueError(
                f"{location}/source_objects differs from the summary protocol"
            )
        if require_held_out and not _parse_bool(
            row["held_out_topology"],
            field="held_out_topology",
            location=location,
        ):
            raise ValueError(f"{location}/held_out_topology must be True")
    return len(rows)


def verify_contact_posterior_provenance_bundle(
    bundle_directory: str | Path,
) -> dict[str, Any]:
    """Verify source integrity plus exact held-out topology provenance."""

    source_integrity = verify_contact_posterior_source_bundle(bundle_directory)
    supplied = Path(bundle_directory)
    if supplied.is_symlink():
        raise ValueError("source bundle directory must not be a symlink")
    if not supplied.is_dir():
        raise FileNotFoundError(f"source bundle directory is missing: {supplied}")
    bundle = supplied.resolve()

    summary = _read_json_object(bundle / "summary.json")
    if summary.get("benchmark") != _BENCHMARK:
        raise ValueError("summary benchmark is unsupported")
    protocol = _read_json_value(bundle / "protocol.json")
    expected_sources = _expected_protocol_sources(summary, protocol)

    recovery_row_count = _validate_topology_rows(
        bundle / "contact_recovery.csv",
        required=_RECOVERY_REQUIRED_FIELDS,
        expected_sources=expected_sources,
        require_held_out=True,
    )
    intervention_row_count = _validate_topology_rows(
        bundle / "interventions.csv",
        required=_INTERVENTION_REQUIRED_FIELDS,
        expected_sources=expected_sources,
        require_held_out=True,
    )
    fold_row_count = _validate_topology_rows(
        bundle / "fold_calibration.csv",
        required=_FOLD_REQUIRED_FIELDS,
        expected_sources=expected_sources,
        require_held_out=False,
    )

    manifest_sha256 = _sha256(bundle / "manifest.json")
    if source_integrity.get("manifest_sha256") != manifest_sha256:
        raise ValueError(
            "source-integrity and provenance verifiers disagree on manifest identity"
        )

    report = dict(source_integrity)
    report["provenance_contract"] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactPosteriorProvenance",
        "manifest_sha256": manifest_sha256,
        "protocol_object_count": len(expected_sources),
        "contact_recovery_row_count": recovery_row_count,
        "intervention_row_count": intervention_row_count,
        "fold_calibration_row_count": fold_row_count,
        "nonfinite_json_rejected": True,
        "protocol_payload_bound": True,
        "source_topology_identities_bound": True,
        "held_out_declarations_bound": True,
        "passed": True,
    }
    report["passed"] = True
    return report
