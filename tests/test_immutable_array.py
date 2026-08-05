from __future__ import annotations

import numpy as np
import pytest

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array


def _assert_irreversibly_read_only(values: np.ndarray) -> None:
    assert not values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        values[...] = 0
    with pytest.raises(ValueError, match="WRITEABLE"):
        values.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        values.flags.writeable = True


def test_readonly_array_owns_values_and_preserves_identity() -> None:
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2)
    expected = np.asarray(source, dtype=np.float64).copy()
    expected_identity = array_sha256(expected)

    frozen = readonly_array(source, dtype=np.float64)
    source[...] = -1

    assert frozen.dtype == expected.dtype
    assert frozen.shape == expected.shape
    np.testing.assert_array_equal(frozen, expected)
    assert array_sha256(frozen) == expected_identity
    _assert_irreversibly_read_only(frozen)


def test_readonly_array_handles_scalar_and_empty_shapes() -> None:
    scalar = readonly_array(4.5)
    empty = readonly_array(np.empty((0, 3), dtype=np.float32))

    assert scalar.shape == ()
    assert empty.shape == (0, 3)
    _assert_irreversibly_read_only(scalar)
    _assert_irreversibly_read_only(empty)


def test_readonly_array_rejects_python_object_storage() -> None:
    values = np.asarray([object()], dtype=object)
    with pytest.raises(ValueError, match="Python objects"):
        readonly_array(values)


def test_readonly_integer_array_rejects_coercion_and_is_irreversible() -> None:
    source = np.asarray([0, 2, 4], dtype=np.uint32)
    frozen = readonly_integer_array(source, name="indices")
    source[...] = 9

    assert frozen.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(frozen, [0, 2, 4])
    _assert_irreversibly_read_only(frozen)

    for invalid in ([1.0], [True], ["1"]):
        with pytest.raises(ValueError, match="must contain integers"):
            readonly_integer_array(invalid, name="indices")


def test_source_does_not_freeze_arrays_by_clearing_only_the_write_flag() -> None:
    import ast
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for source_root in (
        repository_root / "src" / "causal4d",
        repository_root / "src" / "causal4d_public",
    ):
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr != "setflags":
                        continue
                    clears_write = any(
                        keyword.arg == "write"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False
                        for keyword in node.keywords
                    )
                    if clears_write:
                        violations.append(
                            f"{path}:{node.lineno}: setflags(write=False)"
                        )
                if not isinstance(node, ast.Assign):
                    continue
                target = node.targets[0] if len(node.targets) == 1 else None
                if not (
                    isinstance(target, ast.Attribute)
                    and target.attr == "writeable"
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "flags"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is False
                ):
                    continue
                violations.append(f"{path}:{node.lineno}: flags.writeable = False")
    assert not violations, (
        "write flags can be re-enabled; use causal4d.immutable_array.readonly_array:\n"
        + "\n".join(violations)
    )
