from __future__ import annotations

from pathlib import Path

import pytest

import causal4d.atomic_io as atomic_io_module
from causal4d.registered_real_report_shell import main
from tests.test_registered_real_report_shell import _analysis, _analysis_bytes


def test_no_overwrite_pair_rolls_back_json_directory_sync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis()
    analysis_path = tmp_path / "registered-analysis.json"
    analysis_path.write_bytes(_analysis_bytes(analysis))
    shell_path = tmp_path / "report-shell.json"
    markdown_path = tmp_path / "report-shell.md"
    real_fsync_directory = atomic_io_module._fsync_directory
    calls = 0

    def fail_json_directory_sync(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated JSON directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        atomic_io_module,
        "_fsync_directory",
        fail_json_directory_sync,
    )
    with pytest.raises(OSError, match="simulated JSON directory fsync failure"):
        main(
            [
                "render",
                str(analysis_path),
                "--output-json",
                str(shell_path),
                "--output-markdown",
                str(markdown_path),
            ]
        )

    assert calls == 3
    assert not shell_path.exists()
    assert not markdown_path.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))
