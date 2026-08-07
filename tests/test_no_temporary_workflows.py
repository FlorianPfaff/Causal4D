from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = ROOT / ".github" / "workflows"


def test_no_temporary_workflow_can_reach_a_mergeable_head() -> None:
    temporary = sorted(
        path.relative_to(ROOT).as_posix()
        for pattern in ("temporary-*.yml", "temporary-*.yaml")
        for path in WORKFLOW_DIRECTORY.glob(pattern)
    )
    assert temporary == [], (
        "temporary workflows must delete themselves before review and merge: "
        f"{temporary}"
    )
