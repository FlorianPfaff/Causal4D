"""Content-addressed beta-zero physical evaluation records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from causal4d._held_out_target_contract import (
    PAYLOAD_FIELDS,
    SOURCE_FIELDS,
    canonical_json,
    reject_duplicate_json_keys,
    reject_nonfinite_json_constant,
    require_exact_fields,
    require_finite_number,
    require_integer,
    require_integer_interval,
    require_mapping,
    require_nonempty_string,
    require_optional_string,
    validate_sha256,
    validate_target_descriptor,
)
from causal4d.atomic_io import atomic_write_binary
from causal4d.contracts import CausalContext, PhysicalPosterior
from causal4d.held_out_target import HeldOutPhysicalTarget
from causal4d.immutable_json import plain_json, validated_json_mapping

PHYSICAL_COUNTERFACTUAL_EVALUATION_SCHEMA = (
    "causal4d.physical_counterfactual_evaluation"
)
PHYSICAL_COUNTERFACTUAL_EVALUATION_SCHEMA_VERSION = 1

_EVALUATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "evaluation_id",
        "physical_posterior_id",
        "held_out_target_id",
        "held_out_target_descriptor",
        "source_query_id",
        "protocol_id",
        "case",
        "counterfactual_action_id",
        "causal_context",
        "target_frame_interval",
        "evaluation_frame_interval",
        "evaluation_frame_interval_absolute",
        "confidence_level",
        "target_source",
        "target_payload",
        "claim_boundary",
        "semantic_beta",
        "semantic_evidence_consumed",
        "molmo_motion_consumed",
        "valid_point_frames",
        "coordinate_rmse_m",
        "track_error_m",
        "fde_m",
        "coverage",
        "coverage_error",
        "mean_interval_width_m",
        "nees",
        "gaussian_nll",
    }
)


def build_physical_counterfactual_evaluation_record(
    posterior: PhysicalPosterior,
    target: HeldOutPhysicalTarget,
    metrics: Mapping[str, Any],
    *,
    start_frame: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Bind finite beta-zero metrics to exact posterior and target identities."""

    target.require_compatible_physical_posterior(posterior)
    start = require_integer(start_frame, name="start_frame")
    if start >= len(target.positions_m):
        raise ValueError("start_frame must lie inside the held-out target")
    confidence = require_finite_number(
        confidence_level,
        name="confidence_level",
        minimum=0.0,
        maximum=1.0,
    )
    if confidence in {0.0, 1.0}:
        raise ValueError("confidence_level must lie strictly inside (0, 1)")

    normalized_metrics = validated_json_mapping(
        metrics,
        error_message="physical evaluation metrics must be finite JSON data",
    )
    if normalized_metrics.get("physical_posterior_id") != posterior.artifact_id:
        raise ValueError("metrics identify a different physical posterior")
    semantic_beta = require_finite_number(
        normalized_metrics.get("semantic_beta"),
        name="semantic_beta",
    )
    if semantic_beta != 0.0:
        raise ValueError("physical evaluation must retain semantic_beta=0")
    if normalized_metrics.get("semantic_evidence_consumed") is not False:
        raise ValueError("physical evaluation must not consume semantic evidence")

    record: dict[str, Any] = plain_json(normalized_metrics)
    record.update(
        {
            "schema": PHYSICAL_COUNTERFACTUAL_EVALUATION_SCHEMA,
            "schema_version": PHYSICAL_COUNTERFACTUAL_EVALUATION_SCHEMA_VERSION,
            "held_out_target_id": target.artifact_id,
            "held_out_target_descriptor": target.descriptor(),
            "source_query_id": target.source_query_id,
            "protocol_id": target.context.protocol_id,
            "case": target.context.case_id,
            "counterfactual_action_id": target.context.u_cf.action_id,
            "causal_context": target.context.as_dict(),
            "target_frame_interval": [
                target.trajectory_frame_start,
                target.trajectory_frame_stop,
            ],
            "evaluation_frame_interval_absolute": [
                target.trajectory_frame_start + start,
                target.trajectory_frame_stop,
            ],
            "confidence_level": confidence,
            "target_source": target.source,
            "target_payload": target.payload_hashes(),
            "claim_boundary": (
                "beta-zero physical evaluation; no semantic evidence consumed"
            ),
        }
    )
    encoded = canonical_json(record).encode("utf-8")
    record["evaluation_id"] = hashlib.sha256(encoded).hexdigest()
    return validate_physical_counterfactual_evaluation_record(record)


def validate_physical_counterfactual_evaluation_record(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an evaluation record and its content identity."""

    fields = require_exact_fields(
        values,
        name="physical counterfactual evaluation",
        required=_EVALUATION_FIELDS,
    )
    if fields["schema"] != PHYSICAL_COUNTERFACTUAL_EVALUATION_SCHEMA:
        raise ValueError("unsupported physical evaluation schema")
    schema_version = require_integer(
        fields["schema_version"],
        name="physical evaluation schema_version",
        minimum=1,
    )
    if schema_version != PHYSICAL_COUNTERFACTUAL_EVALUATION_SCHEMA_VERSION:
        raise ValueError("unsupported physical evaluation schema version")
    for name in (
        "evaluation_id",
        "physical_posterior_id",
        "held_out_target_id",
        "source_query_id",
    ):
        validate_sha256(fields[name], name=name)

    context = CausalContext.from_dict(fields["causal_context"])
    target_descriptor = validate_target_descriptor(
        fields["held_out_target_descriptor"]
    )
    if target_descriptor["artifact_id"] != fields["held_out_target_id"]:
        raise ValueError("held-out target descriptor identity is inconsistent")
    if target_descriptor["context"] != context.as_dict():
        raise ValueError("held-out target descriptor context is inconsistent")
    if target_descriptor["source_query_id"] != fields["source_query_id"]:
        raise ValueError("held-out target descriptor query identity is inconsistent")
    if fields["protocol_id"] != context.protocol_id:
        raise ValueError("evaluation protocol_id disagrees with causal_context")
    if fields["case"] != context.case_id:
        raise ValueError("evaluation case disagrees with causal_context")
    if fields["counterfactual_action_id"] != context.u_cf.action_id:
        raise ValueError(
            "evaluation counterfactual_action_id disagrees with causal_context"
        )

    target_interval = require_integer_interval(
        fields["target_frame_interval"],
        name="target_frame_interval",
    )
    if target_descriptor["trajectory_frame_interval"] != target_interval:
        raise ValueError("held-out target descriptor interval is inconsistent")
    relative_interval = require_integer_interval(
        fields["evaluation_frame_interval"],
        name="evaluation_frame_interval",
    )
    absolute_interval = require_integer_interval(
        fields["evaluation_frame_interval_absolute"],
        name="evaluation_frame_interval_absolute",
    )
    if relative_interval[1] != target_interval[1] - target_interval[0]:
        raise ValueError("relative evaluation stop does not match target length")
    if absolute_interval != [
        target_interval[0] + relative_interval[0],
        target_interval[1],
    ]:
        raise ValueError("absolute evaluation interval is inconsistent")

    confidence = require_finite_number(
        fields["confidence_level"],
        name="confidence_level",
        minimum=0.0,
        maximum=1.0,
    )
    if confidence in {0.0, 1.0}:
        raise ValueError("confidence_level must lie strictly inside (0, 1)")
    semantic_beta = require_finite_number(
        fields["semantic_beta"],
        name="semantic_beta",
    )
    if semantic_beta != 0.0:
        raise ValueError("physical evaluation must retain semantic_beta=0")
    if fields["semantic_evidence_consumed"] is not False:
        raise ValueError("physical evaluation must not consume semantic evidence")
    if fields["molmo_motion_consumed"] is not False:
        raise ValueError("physical evaluation must not consume MolmoMotion")
    if (
        type(fields["valid_point_frames"]) is not int
        or fields["valid_point_frames"] < 1
    ):
        raise ValueError("valid_point_frames must be a positive integer")
    for name in (
        "coordinate_rmse_m",
        "track_error_m",
        "coverage_error",
        "mean_interval_width_m",
        "nees",
    ):
        require_finite_number(fields[name], name=name, minimum=0.0)
    require_finite_number(fields["gaussian_nll"], name="gaussian_nll")
    require_finite_number(
        fields["coverage"],
        name="coverage",
        minimum=0.0,
        maximum=1.0,
    )
    if fields["fde_m"] is not None:
        require_finite_number(fields["fde_m"], name="fde_m", minimum=0.0)

    source = require_exact_fields(
        fields["target_source"],
        name="target_source",
        required=SOURCE_FIELDS,
    )
    require_nonempty_string(source["kind"], name="target_source.kind")
    require_nonempty_string(source["revision"], name="target_source.revision")
    validate_sha256(
        source["content_sha256"],
        name="target_source.content_sha256",
    )
    require_optional_string(
        source["artifact_id"],
        name="target_source.artifact_id",
    )
    target_payload = require_exact_fields(
        fields["target_payload"],
        name="target_payload",
        required=PAYLOAD_FIELDS,
    )
    for name, value in target_payload.items():
        validate_sha256(value, name=f"target_payload.{name}")
    if target_descriptor["source"] != dict(source):
        raise ValueError("held-out target descriptor source is inconsistent")
    if target_descriptor["payload"] != dict(target_payload):
        raise ValueError("held-out target descriptor payload is inconsistent")
    if fields["claim_boundary"] != (
        "beta-zero physical evaluation; no semantic evidence consumed"
    ):
        raise ValueError("unexpected physical evaluation claim boundary")

    normalized = plain_json(
        validated_json_mapping(
            fields,
            error_message="physical evaluation must be finite JSON data",
        )
    )
    supplied_id = normalized.pop("evaluation_id")
    expected_id = hashlib.sha256(
        canonical_json(normalized).encode("utf-8")
    ).hexdigest()
    if supplied_id != expected_id:
        raise ValueError("physical evaluation_id does not match its contents")
    normalized["evaluation_id"] = supplied_id
    return normalized


def load_physical_counterfactual_evaluation_record(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate one claim-bearing physical evaluation JSON."""

    try:
        parsed = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_json_keys,
            parse_constant=reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("physical evaluation JSON is invalid") from error
    return validate_physical_counterfactual_evaluation_record(
        require_mapping(parsed, name="physical evaluation JSON")
    )


def save_physical_counterfactual_evaluation_record(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    """Atomically publish and independently reload a physical evaluation."""

    validated = validate_physical_counterfactual_evaluation_record(record)
    serialized = (
        json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    expected_id = validated["evaluation_id"]

    def write(handle: BinaryIO) -> None:
        handle.write(serialized)

    def validate(temporary: Path) -> None:
        restored = load_physical_counterfactual_evaluation_record(temporary)
        if restored["evaluation_id"] != expected_id:
            raise ValueError("physical evaluation changed during serialization")

    atomic_write_binary(
        path,
        write,
        overwrite=overwrite,
        validate=validate,
    )


__all__ = [
    "PHYSICAL_COUNTERFACTUAL_EVALUATION_SCHEMA",
    "PHYSICAL_COUNTERFACTUAL_EVALUATION_SCHEMA_VERSION",
    "build_physical_counterfactual_evaluation_record",
    "load_physical_counterfactual_evaluation_record",
    "save_physical_counterfactual_evaluation_record",
    "validate_physical_counterfactual_evaluation_record",
]
