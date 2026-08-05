"""Deeply immutable, finite JSON values for content-addressed artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Any, NoReturn, overload


class _FrozenJSONMapping(Mapping[str, Any]):
    """An immutable JSON object without a mutable ``dict`` base class."""

    __slots__ = ("__values",)
    __values: Mapping[str, Any]

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(
            self,
            "_FrozenJSONMapping__values",
            MappingProxyType(dict(values)),
        )

    def __getitem__(self, key: str) -> Any:
        return self.__values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__values)

    def __len__(self) -> int:
        return len(self.__values)

    def __repr__(self) -> str:
        return repr(dict(self.__values))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise TypeError("JSON data is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        del name
        raise TypeError("JSON data is immutable")

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("JSON data is immutable")

    def __setitem__(self, key: object, value: object) -> NoReturn:
        self._immutable(key, value)

    def __delitem__(self, key: object) -> NoReturn:
        self._immutable(key)

    def clear(self) -> NoReturn:
        self._immutable()

    def pop(self, *args: object) -> NoReturn:
        self._immutable(*args)

    def popitem(self) -> NoReturn:
        self._immutable()

    def setdefault(self, *args: object) -> NoReturn:
        self._immutable(*args)

    def update(self, *args: object, **kwargs: object) -> NoReturn:
        self._immutable(*args, **kwargs)

    def __ior__(self, other: object) -> NoReturn:
        self._immutable(other)

    def __copy__(self) -> dict[str, Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        del memo
        return plain_json(self)


class _FrozenJSONSequence(Sequence[Any]):
    """An immutable JSON array without a mutable ``list`` base class."""

    __slots__ = ("__values",)
    __values: tuple[Any, ...]

    def __init__(self, values: Sequence[Any]) -> None:
        object.__setattr__(self, "_FrozenJSONSequence__values", tuple(values))

    @overload
    def __getitem__(self, index: int) -> Any: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Any, ...]: ...

    def __getitem__(self, index: int | slice) -> Any | tuple[Any, ...]:
        return self.__values[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__values)

    def __len__(self) -> int:
        return len(self.__values)

    def __repr__(self) -> str:
        return repr(list(self.__values))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenJSONSequence):
            return self.__values == other.__values
        if isinstance(other, (list, tuple)):
            return self.__values == tuple(other)
        return False

    def __setattr__(self, name: str, value: object) -> NoReturn:
        del name, value
        raise TypeError("JSON data is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        del name
        raise TypeError("JSON data is immutable")

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("JSON data is immutable")

    def __setitem__(self, key: object, value: object) -> NoReturn:
        self._immutable(key, value)

    def __delitem__(self, key: object) -> NoReturn:
        self._immutable(key)

    def append(self, value: object) -> NoReturn:
        self._immutable(value)

    def clear(self) -> NoReturn:
        self._immutable()

    def extend(self, values: object) -> NoReturn:
        self._immutable(values)

    def insert(self, index: object, value: object) -> NoReturn:
        self._immutable(index, value)

    def pop(self, *args: object) -> NoReturn:
        self._immutable(*args)

    def remove(self, value: object) -> NoReturn:
        self._immutable(value)

    def reverse(self) -> NoReturn:
        self._immutable()

    def sort(self, *args: object, **kwargs: object) -> NoReturn:
        self._immutable(*args, **kwargs)

    def __iadd__(self, other: object) -> NoReturn:
        self._immutable(other)

    def __imul__(self, other: object) -> NoReturn:
        self._immutable(other)

    def __copy__(self) -> list[Any]:
        return plain_json(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        del memo
        return plain_json(self)


_JSON_ARRAY_TYPES = (list, tuple, _FrozenJSONSequence)


def plain_json(value: Any) -> Any:
    """Return ordinary mutable ``dict``/``list`` JSON containers recursively."""

    if isinstance(value, Mapping):
        return {key: plain_json(item) for key, item in value.items()}
    if isinstance(value, _JSON_ARRAY_TYPES):
        return [plain_json(item) for item in value]
    return value


def _require_string_mapping_keys(value: Any) -> None:
    """Reject JSON objects whose Python keys would be coerced by ``json``."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _require_string_mapping_keys(item)
    elif isinstance(value, _JSON_ARRAY_TYPES):
        for item in value:
            _require_string_mapping_keys(item)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenJSONMapping(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return _FrozenJSONSequence(tuple(_freeze_json(item) for item in value))
    return value


def validated_json_mapping(
    values: Mapping[str, Any],
    *,
    error_message: str = "metadata must be finite JSON data",
) -> Mapping[str, Any]:
    """Normalize a mapping as finite JSON and recursively freeze its containers.

    Tuples become JSON arrays. Object keys must already be strings so the JSON
    encoder cannot silently coerce a key or collapse distinct Python identities.
    Non-finite and unsupported values fail closed. The returned values implement
    the read-only ``Mapping`` and ``Sequence`` protocols without inheriting from
    mutable ``dict`` or ``list``. Call :func:`plain_json` at serialization or
    explicit mutable-export boundaries.
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
