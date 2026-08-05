#!/usr/bin/env python3
"""Find one exact Deform360 derived-data root on a research runner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


_REQUIRED = (
    Path("aligned/002-rope-silk/episode_0000/robot/robot.npz"),
    Path("observations/002-rope-silk/episode_0000/sampled_hulls.json"),
    Path("aligned/170-spider/episode_0000/robot/robot.npz"),
    Path("observations/170-spider/episode_0000/sampled_hulls.json"),
)


def _valid(root: Path) -> bool:
    return root.is_dir() and all((root / path).is_file() for path in _REQUIRED)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="*", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configured = os.environ.get("DEFORM360_REPLICATION_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _valid(root):
            raise SystemExit(
                "DEFORM360_REPLICATION_ROOT does not contain the locked "
                f"derived dataset: {root}"
            )
        print(root)
        return

    candidates = list(args.candidates)
    candidates.extend(
        [
            Path("/home/florianpfaff/codex-runs/deform360-replication-locked-v1"),
            Path("/home/github-runner/.cache/datasets/deform360"),
            Path("/home/github-runner/.cache/datasets/deform360/derived"),
        ]
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if _valid(root):
            print(root)
            return

    matches: dict[Path, None] = {}
    for parent in (
        Path("/home/github-runner/.cache/datasets"),
        Path("/home/florianpfaff/codex-runs"),
    ):
        if not parent.is_dir():
            continue
        for hull in parent.glob(
            "**/observations/002-rope-silk/episode_0000/sampled_hulls.json"
        ):
            root = hull.parents[3].resolve()
            if _valid(root):
                matches[root] = None
    if len(matches) != 1:
        rendered = ", ".join(map(str, sorted(matches))) or "none"
        raise SystemExit(
            "expected one discoverable Deform360 replication root; "
            f"found {rendered}. Set DEFORM360_REPLICATION_ROOT explicitly."
        )
    print(next(iter(matches)))


if __name__ == "__main__":
    main()
