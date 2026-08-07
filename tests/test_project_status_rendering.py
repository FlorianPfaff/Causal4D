from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CI_ROOT = ROOT / "ci"
STATUS = CI_ROOT / "project_status_v2.json"
RENDERED = ROOT / "docs/current_project_status.md"
if str(CI_ROOT) not in sys.path:
    sys.path.insert(0, str(CI_ROOT))

from render_project_status import render_project_status  # noqa: E402


def test_rendered_project_status_is_current() -> None:
    rendered = render_project_status(STATUS)

    assert RENDERED.read_text(encoding="utf-8") == rendered
    assert "0/36 acquired; 0/36 validated" in rendered
    assert "controlled synthetic | `passed`" in rendered
    assert "fresh real provider | `pending`" in rendered


def test_renderer_rejects_historical_schema(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="schema version 2"):
        render_project_status(CI_ROOT / "project_status_v1.json")
