"""Deeply immutable, finite JSON values for content-addressed artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def plain_json(value: Any) -> Any:
    """Return ordinary mutable ``dict``/``list`` JSON containers recursively."""

    if isinstance(value, Mapping):
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_json(item) for item in value]
    return value


def _require_string_mapping_keys(value: Any) -> None:
    """Reject JSON objects whose Python keys would be coerced by ``json``."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _require_string_mapping_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_string_mapping_keys(item)


class _FrozenDict(dict):
    """A JSON object that preserves ``dict`` compatibility but rejects mutation."""

    __slots__ = ()
    _MUTATORS = frozenset({"clear", "pop", "popitem", "setdefault", "update"})

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._MUTATORS:
            return self._immutable
        return super().__getattribute__(name)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("JSON data is immutable")

    def __setitem__(self, key: object, value: object) -> None:
        self._immutable(key, value)

    def __delitem__(self, key: object) -> None:
        self._immutable(key)

    def __ior__(self, other: object) -> _FrozenDict:  # type: ignore[misc]
        self._immutable(other)
        return self

    def __copy__(self) -> dict[str, Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        del memo
        return plain_json(self)


class _FrozenList(list):
    """A JSON array that preserves ``list`` compatibility but rejects mutation."""

    __slots__ = ()
    _MUTATORS = frozenset(
        {"append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort"}
    )

    def __getattribute__(self, name: str) -> Any:
        if name in type(self)._MUTATORS:
            return self._immutable
        return super().__getattribute__(name)

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("JSON data is immutable")

    def __setitem__(self, key: object, value: object) -> None:
        self._immutable(key, value)

    def __delitem__(self, key: object) -> None:
        self._immutable(key)

    def __iadd__(self, other: object) -> _FrozenList:  # type: ignore[misc]
        self._immutable(other)
        return self

    def __imul__(self, other: object) -> _FrozenList:
        self._immutable(other)
        return self

    def __copy__(self) -> list[Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        del memo
        return plain_json(self)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze_json(item) for item in value)
    return value


def validated_json_mapping(
    values: Mapping[str, Any],
    *,
    error_message: str = "metadata must be finite JSON data",
) -> Mapping[str, Any]:
    """Normalize a mapping as finite JSON and recursively freeze its containers.

    Tuples become JSON arrays. Object keys must already be strings so the JSON
    encoder cannot silently coerce a key or collapse distinct Python identities.
    Non-finite and unsupported values fail closed. The returned containers still
    satisfy ordinary ``isinstance(value, dict/list)`` checks, while ``copy.copy``
    and ``copy.deepcopy`` yield independent mutable JSON data.
    """

    try:
        _require_string_mapping_keys(values)
        normalized = json.loads(
            json.dumps(plain_json(values), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(error_message) from error
    if not isinstance(normalized, dict):
        raise ValueError(error_message)
    return _freeze_json(normalized)


__all__ = ["plain_json", "validated_json_mapping"]
