from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(
            f"expected exactly one patch anchor in {path}: {old[:120]!r}; "
            f"found {text.count(old)}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


prefix = Path("src/causal4d_public/deform360_prefix_kinematics.py")
replace_once(
    prefix,
    '''    raw_indices = association.get("selected_taxel_indices")
    _require(
        isinstance(raw_indices, Sequence)
        and not isinstance(raw_indices, (str, bytes))
        and len(raw_indices) >= 1,
        "contact association has no taxel indices",
    )
    _require(
        all(type(index) is int and index >= 0 for index in raw_indices),
        "contact association taxel indices must be nonnegative integers",
    )
''',
    '''    raw_indices = association.get("selected_taxel_indices")
    if (
        not isinstance(raw_indices, Sequence)
        or isinstance(raw_indices, (str, bytes))
        or len(raw_indices) < 1
    ):
        raise ValueError("contact association has no taxel indices")
    _require(
        all(type(index) is int and index >= 0 for index in raw_indices),
        "contact association taxel indices must be nonnegative integers",
    )
''',
)

diagnostic = Path(
    "src/causal4d_public/deform360_prefix_kinematics_diagnostic.py"
)
replace_once(
    diagnostic,
    '''        source_path = entry.get("source_path")
        _require(
            type(source_path) is str and source_path not in seen,
            "source manifest path is invalid or repeated",
        )
        seen.add(source_path)
        path = root / source_path
''',
    '''        source_path = entry.get("source_path")
        if (
            type(source_path) is not str
            or not source_path
            or source_path in seen
        ):
            raise ValueError("source manifest path is invalid or repeated")
        seen.add(source_path)
        path = root / source_path
''',
)
