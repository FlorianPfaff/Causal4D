"""Run MolmoMotion with digest-complete PhysTwin visual inputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    """Load optional integrations only after argparse handles ``--help``."""

    global prepare_molmo_phystwin_query_v2
    global run_molmo_motion_forecasts_v2
    global save_molmo_forecasts

    from causal4d.molmo_adapter import save_molmo_forecasts
    from causal4d.molmo_visual_inputs_v2 import (
        prepare_molmo_phystwin_query_v2,
        run_molmo_motion_forecasts_v2,
    )


def _caption(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("captions must use ID=TEXT")
    identifier, text = value.split("=", 1)
    if not identifier or not text.strip():
        raise argparse.ArgumentTypeError("caption id and text must be nonempty")
    return identifier, text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast exact PhysTwin material points using a digest-complete "
            "visual-input manifest."
        )
    )
    parser.add_argument("final_data")
    parser.add_argument("raw_case_dir")
    parser.add_argument("visual_input_manifest")
    parser.add_argument("checkpoint")
    parser.add_argument("output_npz")
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--point-count", type=int, default=8)
    parser.add_argument("--future-horizon", type=int, default=30)
    parser.add_argument(
        "--forecast-fps",
        type=float,
        default=15.0,
        help="Molmo timestamp rate; the released H3/F30 checkpoint uses 15 fps",
    )
    parser.add_argument("--camera-index", type=int)
    parser.add_argument(
        "--caption",
        type=_caption,
        action="append",
        required=True,
        help="repeat ID=TEXT for instruction and language controls",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    captions = dict(args.caption)
    if len(captions) != len(args.caption):
        raise ValueError("caption ids must be unique")
    query = prepare_molmo_phystwin_query_v2(
        args.final_data,
        args.raw_case_dir,
        args.visual_input_manifest,
        train_end_frame=args.train_end_frame,
        history_size=args.history_size,
        point_count=args.point_count,
        camera_index=args.camera_index,
        forecast_fps=args.forecast_fps,
    )
    bundle = run_molmo_motion_forecasts_v2(
        query,
        args.checkpoint,
        captions,
        future_horizon=args.future_horizon,
    )
    save_molmo_forecasts(args.output_npz, bundle)
    print(
        json.dumps(
            {
                "output": str(Path(args.output_npz).resolve()),
                "case": query.case_name,
                "camera": query.camera_index,
                "node_indices": query.node_indices.tolist(),
                "forecast_ids": list(bundle.forecast_ids),
                "future_shape": list(bundle.future_world_m.shape),
                "source_fps": query.source_fps,
                "forecast_fps": query.forecast_fps,
                "frame_stride": query.frame_stride,
                "visual_input_artifact_id": query.visual_input_artifact_id,
                "visual_input_manifest_sha256": (query.visual_input_manifest_sha256),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
