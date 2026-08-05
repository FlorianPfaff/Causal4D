import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from causal4d.immutable_json import plain_json, validated_json_mapping


def test_validated_json_mapping_is_deeply_immutable_and_copyable() -> None:
    source = {
        "nested": {
            "items": [1, {"accepted": True}],
            "tuple_items": (2, 3),
        }
    }
    frozen = validated_json_mapping(source)

    source["nested"]["items"][1]["accepted"] = False
    assert frozen["nested"]["items"][1]["accepted"] is True
    assert frozen["nested"]["tuple_items"] == [2, 3]
    assert isinstance(frozen, Mapping)
    assert isinstance(frozen["nested"]["items"], Sequence)
    assert not isinstance(frozen, dict)
    assert not isinstance(frozen["nested"]["items"], list)

    with pytest.raises(TypeError, match="immutable"):
        frozen["new"] = "value"
    with pytest.raises(TypeError, match="immutable"):
        frozen["nested"]["items"][1]["accepted"] = False
    with pytest.raises(TypeError, match="immutable"):
        frozen.update({"new": "value"})
    with pytest.raises(TypeError, match="immutable"):
        frozen["nested"]["items"].append("value")
    with pytest.raises(TypeError, match="immutable"):
        frozen["nested"]["items"] += ["value"]

    shallow = copy.copy(frozen)
    deep = copy.deepcopy(frozen)
    assert type(shallow) is dict
    assert type(deep) is dict
    assert type(deep["nested"]["items"]) is list
    deep["nested"]["items"].append("copy-only")
    assert "copy-only" not in frozen["nested"]["items"]
    assert plain_json(frozen) == shallow
    assert deep == {
        "nested": {
            "items": [1, {"accepted": True}, "copy-only"],
            "tuple_items": [2, 3],
        }
    }


def test_validated_json_mapping_blocks_mutable_builtin_base_class_bypasses() -> None:
    frozen = validated_json_mapping({"nested": {"items": [1, 2]}})
    nested = frozen["nested"]
    items = nested["items"]
    expected = plain_json(frozen)

    with pytest.raises(TypeError):
        dict.__setitem__(frozen, "changed", True)
    with pytest.raises(TypeError):
        dict.update(nested, {"changed": True})
    with pytest.raises(TypeError):
        list.append(items, 3)
    with pytest.raises(TypeError):
        list.__setitem__(items, 0, 99)

    assert plain_json(frozen) == expected


def test_frozen_json_requires_explicit_plain_export_for_serialization() -> None:
    frozen = validated_json_mapping({"nested": {"items": [1, 2]}})

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(frozen)
    assert json.loads(json.dumps(plain_json(frozen), sort_keys=True)) == {
        "nested": {"items": [1, 2]}
    }


def test_validated_json_mapping_rejects_nonfinite_and_non_json_values() -> None:
    with pytest.raises(ValueError, match="finite JSON"):
        validated_json_mapping({"bad": float("nan")})
    with pytest.raises(ValueError, match="finite JSON"):
        validated_json_mapping({"bad": object()})


def test_validated_json_mapping_rejects_non_string_object_keys() -> None:
    top_level: dict[Any, Any] = {1: "integer-key"}
    nested: dict[Any, Any] = {"valid": [{False: "boolean-key"}]}
    colliding_after_json_coercion: dict[Any, Any] = {
        1: "integer-key",
        "1": "string-key",
    }

    for values in (top_level, nested, colliding_after_json_coercion):
        with pytest.raises(ValueError, match="finite JSON"):
            validated_json_mapping(values)
