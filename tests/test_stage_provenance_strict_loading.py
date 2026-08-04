from copy import deepcopy

import pytest

from causal4d.stage_provenance import (
    CounterfactualQueryContext,
    EvaluationTarget,
    FactualEvidenceContext,
)


def _observation_window(*, start: int, stop: int) -> dict[str, object]:
    return {
        "case_id": "strict-case",
        "stream_id": "object_points_m",
        "frame_start": start,
        "frame_stop": stop,
        "content_sha256": "a" * 64,
    }


def _action_window(*, start: int, stop: int) -> dict[str, object]:
    return {
        "action_id": "u",
        "case_id": "strict-case",
        "frame_start": start,
        "frame_stop": stop,
        "trajectory_sha256": "b" * 64,
        "provenance": "unit test",
    }


def _query_values() -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract_type": "CounterfactualQueryContext",
        "protocol_id": "strict-protocol",
        "case_id": "strict-case",
        "u_cf": _action_window(start=4, stop=8),
        "contact_policy": "new_contact",
        "language": None,
        "query_node_indices": [0, 2],
    }


def test_query_context_rejects_values_that_would_be_coerced() -> None:
    valid = _query_values()
    restored = CounterfactualQueryContext.from_dict(valid)
    assert restored.query_node_indices == (0, 2)

    invalid_values: list[tuple[dict[str, object], str]] = []

    boolean_schema = deepcopy(valid)
    boolean_schema["schema_version"] = True
    invalid_values.append((boolean_schema, "must be an integer"))

    null_protocol = deepcopy(valid)
    null_protocol["protocol_id"] = None
    invalid_values.append((null_protocol, "must be a string"))

    fractional_nodes = deepcopy(valid)
    fractional_nodes["query_node_indices"] = [0, 2.5]
    invalid_values.append((fractional_nodes, "contain only integers"))

    boolean_nodes = deepcopy(valid)
    boolean_nodes["query_node_indices"] = [0, True]
    invalid_values.append((boolean_nodes, "contain only integers"))

    tuple_nodes = deepcopy(valid)
    tuple_nodes["query_node_indices"] = (0, 2)
    invalid_values.append((tuple_nodes, "must be null or a JSON array"))

    fractional_frame = deepcopy(valid)
    assert isinstance(fractional_frame["u_cf"], dict)
    fractional_frame["u_cf"]["frame_start"] = 4.5
    invalid_values.append((fractional_frame, "must be an integer"))

    unknown_field = deepcopy(valid)
    unknown_field["unregistered"] = "value"
    invalid_values.append((unknown_field, "fields do not match the schema"))

    for values, message in invalid_values:
        with pytest.raises(ValueError, match=message):
            CounterfactualQueryContext.from_dict(values)


def test_factual_and_target_contexts_require_exact_nested_schemas() -> None:
    factual = {
        "schema_version": 1,
        "contract_type": "FactualEvidenceContext",
        "protocol_id": "strict-protocol",
        "o_minus": _observation_window(start=0, stop=4),
        "o_plus_prefix": _observation_window(start=4, stop=6),
        "u_obs": _action_window(start=0, stop=8),
    }
    target = {
        "schema_version": 1,
        "contract_type": "EvaluationTarget",
        "protocol_id": "strict-protocol",
        "target": _observation_window(start=6, stop=8),
    }

    assert FactualEvidenceContext.from_dict(factual).case_id == "strict-case"
    assert EvaluationTarget.from_dict(target).case_id == "strict-case"

    malformed_factual = deepcopy(factual)
    assert isinstance(malformed_factual["o_minus"], dict)
    malformed_factual["o_minus"]["frame_start"] = False
    with pytest.raises(ValueError, match="must be an integer"):
        FactualEvidenceContext.from_dict(malformed_factual)

    malformed_target = deepcopy(target)
    assert isinstance(malformed_target["target"], dict)
    malformed_target["target"]["unexpected"] = 1
    with pytest.raises(ValueError, match="fields do not match the schema"):
        EvaluationTarget.from_dict(malformed_target)
