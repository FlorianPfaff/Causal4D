from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = ROOT / ".github" / "workflows"


def test_no_one_shot_workflow_can_reach_a_mergeable_head() -> None:
    one_shot = sorted(
        path.relative_to(ROOT).as_posix()
        for pattern in (
            "temporary-*.yml",
            "temporary-*.yaml",
            "publish-reviewed-*.yml",
            "publish-reviewed-*.yaml",
        )
        for path in WORKFLOW_DIRECTORY.glob(pattern)
    )
    assert one_shot == [], (
        "one-shot workflows must delete themselves before review and merge: "
        f"{one_shot}"
    )
