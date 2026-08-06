"""Create a small synthetic MolmoMotion-style NPZ for the bridge quickstart."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def build_demo_arrays(
    *,
    point_count: int = 8,
    future_count: int = 6,
) -> dict[str, np.ndarray]:
    if point_count < 2:
        raise ValueError("point_count must be at least two")
    if future_count < 1:
        raise ValueError("future_count must be positive")

    columns = int(np.ceil(np.sqrt(point_count)))
    rows = int(np.ceil(point_count / columns))
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.12, 0.12, columns),
        np.linspace(-0.08, 0.08, rows),
    )
    anchor = np.column_stack(
        (
            grid_x.reshape(-1)[:point_count],
            grid_y.reshape(-1)[:point_count],
            np.zeros(point_count),
        )
    )
    time = np.arange(1, future_count + 1, dtype=np.float64) / 15.0
    phase = np.linspace(0.0, np.pi, point_count, dtype=np.float64)

    instruction = np.repeat(anchor[:, None, :], future_count, axis=1)
    instruction[:, :, 2] += 0.10 * time[None]
    instruction[:, :, 2] += (
        0.008 * np.sin(phase)[:, None] * (1.0 - np.exp(-4.0 * time[None]))
    )

    paraphrase = instruction.copy()
    paraphrase[:, :, 0] += 0.002 * np.sin(phase)[:, None] * time[None]

    shuffled = np.repeat(anchor[:, None, :], future_count, axis=1)
    shuffled[:, :, 0] += 0.10 * time[None]

    future = np.stack((instruction, paraphrase, shuffled), axis=0)
    return {
        "node_indices": np.arange(point_count, dtype=np.int64),
        "anchor_positions_world_m": anchor.astype(np.float64),
        "future_positions_world_m": future.astype(np.float64),
        "validity_mask": np.ones(future.shape[:-1], dtype=bool),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a synthetic MolmoMotion-style bridge input NPZ."
    )
    parser.add_argument("output_npz")
    parser.add_argument("--point-count", type=int, default=8)
    parser.add_argument("--future-count", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_npz)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = build_demo_arrays(
        point_count=args.point_count,
        future_count=args.future_count,
    )
    with output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    print(
        json.dumps(
            {
                "forecast_shape_kpfc": list(arrays["future_positions_world_m"].shape),
                "forecast_ids": ["instruction", "paraphrase", "shuffled"],
                "output": str(output.resolve()),
                "point_count": int(len(arrays["node_indices"])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
