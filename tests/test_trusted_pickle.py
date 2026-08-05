from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pytest

from causal4d.trusted_pickle import load_trusted_pickle


def _write_marker(path: str) -> dict[str, bool]:
    Path(path).write_text("executed\n", encoding="utf-8")
    return {"executed": True}


class _ExecutablePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (_write_marker, (str(self.marker),))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pickle_loading_requires_explicit_consent(tmp_path: Path) -> None:
    path = tmp_path / "value.pkl"
    path.write_bytes(pickle.dumps({"value": 4}))
    with pytest.raises(PermissionError, match="pickle loading is disabled"):
        load_trusted_pickle(path)


def test_matching_digest_loads_exact_payload(tmp_path: Path) -> None:
    path = tmp_path / "value.pkl"
    path.write_bytes(pickle.dumps({"value": [1, 2, 3]}))
    assert load_trusted_pickle(
        path,
        allow_unsafe_pickle=True,
        expected_sha256=_digest(path),
    ) == {"value": [1, 2, 3]}


def test_digest_mismatch_rejects_before_unpickling(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    path = tmp_path / "executable.pkl"
    path.write_bytes(pickle.dumps(_ExecutablePayload(marker)))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_trusted_pickle(
            path,
            allow_unsafe_pickle=True,
            expected_sha256="0" * 64,
        )
    assert not marker.exists()


@pytest.mark.parametrize("value", [1, "yes", None])
def test_consent_must_be_an_exact_boolean(tmp_path: Path, value) -> None:
    path = tmp_path / "value.pkl"
    path.write_bytes(pickle.dumps(1))
    with pytest.raises(TypeError, match="exact boolean"):
        load_trusted_pickle(path, allow_unsafe_pickle=value)


def test_digest_must_be_lowercase_hex(tmp_path: Path) -> None:
    path = tmp_path / "value.pkl"
    path.write_bytes(pickle.dumps(1))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_trusted_pickle(
            path,
            allow_unsafe_pickle=True,
            expected_sha256="A" * 64,
        )


def test_symlinked_pickle_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.pkl"
    source.write_bytes(pickle.dumps(1))
    link = tmp_path / "link.pkl"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(ValueError, match="contains a symlink"):
        load_trusted_pickle(link, allow_unsafe_pickle=True)
