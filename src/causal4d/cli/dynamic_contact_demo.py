"""Render a collaborator-facing controlled dynamic-contact demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.cli.dynamic_contact_benchmark import (
    DelayedContactTrace,
    delayed_contact_case,
    delayed_contact_trace,
)
from causal4d.dynamic_contact import ContactRegime


DEMO_SEEDS = tuple(range(10))
DEMO_PREFIX_FRAME_COUNTS = (4, 6, 8, 10)


def aggregate_demo_cases() -> dict[str, Any]:
    """Aggregate the same 40 controlled cases used by the smoke evaluation."""

    cases = [
        delayed_contact_case(seed=seed, prefix_frame_count=prefix_frame_count)
        for seed in DEMO_SEEDS
        for prefix_frame_count in DEMO_PREFIX_FRAME_COUNTS
    ]
    all_gates_passed = all(all(case["gates"].values()) for case in cases)
    return {
        "protocol": "delayed_contact_onset_v1",
        "case_count": len(cases),
        "seeds": list(DEMO_SEEDS),
        "prefix_frame_counts": list(DEMO_PREFIX_FRAME_COUNTS),
        "mean_static_persistence_rmse_mm": float(
            1000.0
            * np.mean([case["static_prefix_persistence_rmse_m"] for case in cases])
        ),
        "mean_dynamic_contact_rmse_mm": float(
            1000.0 * np.mean([case["dynamic_contact_rmse_m"] for case in cases])
        ),
        "mean_relative_rmse_improvement_percent": float(
            100.0 * np.mean([case["relative_rmse_improvement"] for case in cases])
        ),
        "mean_contact_onset_absolute_error_frames": float(
            np.mean([case["contact_onset_absolute_error_frames"] for case in cases])
        ),
        "mean_future_coverage": float(
            np.mean([case["future_coverage"] for case in cases])
        ),
        "future_observations_read": int(
            sum(case["future_observations_read"] for case in cases)
        ),
        "all_gates_passed": all_gates_passed,
        "gate_pass_count": int(sum(all(case["gates"].values()) for case in cases)),
    }


def _summary_markdown(
    aggregate: dict[str, Any],
    representative: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Causal4D controlled delayed-contact demo",
            "",
            "> Controlled simulation. This is mechanism evidence, not real-object validation.",
            "",
            "## Aggregate 40-case result",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            (
                "| Static contact persistence RMSE | "
                f"{aggregate['mean_static_persistence_rmse_mm']:.3f} mm |"
            ),
            (
                "| Dynamic Causal4D contact RMSE | "
                f"{aggregate['mean_dynamic_contact_rmse_mm']:.3f} mm |"
            ),
            (
                "| Relative RMSE improvement | "
                f"{aggregate['mean_relative_rmse_improvement_percent']:.2f}% |"
            ),
            (
                "| Mean contact-onset error | "
                f"{aggregate['mean_contact_onset_absolute_error_frames']:.3f} frames |"
            ),
            (
                "| Registered gates passed | "
                f"{aggregate['gate_pass_count']}/{aggregate['case_count']} |"
            ),
            (f"| Future observations used | {aggregate['future_observations_read']} |"),
            "",
            "## Animated representative case",
            "",
            f"- Seed: `{representative['seed']}`",
            f"- Contact onset: frame `{representative['true_contact_onset_frame']}`",
            (
                "- Posterior expected onset: frame "
                f"`{representative['posterior_expected_contact_onset_frame']:.3f}`"
            ),
            (
                "- Static/dynamic RMSE: "
                f"`{1000.0 * representative['static_prefix_persistence_rmse_m']:.3f}`/"
                f"`{1000.0 * representative['dynamic_contact_rmse_m']:.3f}` mm"
            ),
            "",
            "The video reveals the held-out trajectory only for visualization. The",
            "dynamic posterior itself is inferred from the permitted prefix and action",
            "signal; its recorded future-observation count is zero.",
            "",
        ]
    )


def _render_animation(
    trace: DelayedContactTrace,
    aggregate: dict[str, Any],
    output_dir: Path,
    *,
    fps: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import imageio_ffmpeg
    from matplotlib import animation, pyplot as plt

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    frames = np.arange(trace.truth_m.shape[0])
    truth_x = 1000.0 * trace.truth_m[:, 0, 0]
    static_x = 1000.0 * trace.static_persistence_m[:, 0, 0]
    dynamic_x = 1000.0 * trace.posterior.mean_m[:, 0, 0]
    lower_x = 1000.0 * trace.posterior.interval_lower_m[:, 0, 0]
    upper_x = 1000.0 * trace.posterior.interval_upper_m[:, 0, 0]
    sticking_probability = trace.posterior.regime_probabilities[
        :, ContactRegime.STICKING
    ]
    prefix = int(trace.summary["prefix_frame_count"])

    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#f4f5f3")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.7, 1.0),
        height_ratios=(1.3, 1.0),
        hspace=0.34,
        wspace=0.24,
    )
    trajectory_axis = figure.add_subplot(grid[0, 0])
    contact_axis = figure.add_subplot(grid[1, 0], sharex=trajectory_axis)
    scene_axis = figure.add_subplot(grid[:, 1])

    figure.suptitle(
        "Causal4D infers delayed contact instead of freezing the prefix state",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.925,
        "CONTROLLED SIMULATION | causal prefix only | no future observations used",
        ha="center",
        fontsize=10.5,
        color="#4b5563",
    )

    for axis in (trajectory_axis, contact_axis, scene_axis):
        axis.set_facecolor("#ffffff")
    for axis in (trajectory_axis, contact_axis):
        axis.grid(True, color="#d9dde3", linewidth=0.8, alpha=0.75)
        axis.axvspan(-0.5, prefix - 0.5, color="#1565c0", alpha=0.07)
        axis.axvline(prefix - 0.5, color="#1565c0", linestyle=":", linewidth=2)

    trajectory_axis.fill_between(
        frames,
        lower_x,
        upper_x,
        color="#00897b",
        alpha=0.16,
        label="Causal4D 90% interval",
    )
    trajectory_axis.plot(
        frames,
        truth_x,
        color="#202124",
        linewidth=2.4,
        label="Ground truth",
    )
    trajectory_axis.plot(
        frames,
        static_x,
        color="#c62828",
        linestyle="--",
        linewidth=2.2,
        label="Static contact persistence",
    )
    trajectory_axis.plot(
        frames,
        dynamic_x,
        color="#00897b",
        linewidth=2.5,
        label="Dynamic Causal4D belief",
    )
    truth_marker = trajectory_axis.plot([], [], "o", color="#202124", ms=7)[0]
    static_marker = trajectory_axis.plot([], [], "o", color="#c62828", ms=7)[0]
    dynamic_marker = trajectory_axis.plot([], [], "o", color="#00897b", ms=7)[0]
    trajectory_cursor = trajectory_axis.axvline(0, color="#6b7280", alpha=0.55)
    trajectory_axis.set_ylabel("Material-point displacement (mm)")
    trajectory_axis.set_title("Predicted held-out motion", loc="left", fontsize=12)
    trajectory_axis.legend(loc="upper left", frameon=False, fontsize=9)
    trajectory_axis.set_xlim(-0.5, len(frames) - 0.5)
    trajectory_axis.set_ylim(
        min(-4.0, float(np.min(lower_x)) - 2.0),
        float(np.max(upper_x)) + 7.0,
    )

    contact_axis.plot(
        frames,
        trace.command_activation,
        color="#1565c0",
        linewidth=2.0,
        label="Command activation",
    )
    contact_axis.plot(
        frames,
        sticking_probability,
        color="#7b1fa2",
        linewidth=2.4,
        label="Posterior P(sticking)",
    )
    contact_cursor = contact_axis.axvline(0, color="#6b7280", alpha=0.55)
    contact_marker = contact_axis.plot([], [], "o", color="#7b1fa2", ms=7)[0]
    contact_axis.set_ylim(-0.05, 1.08)
    contact_axis.set_xlabel("Frame")
    contact_axis.set_ylabel("Probability / command")
    contact_axis.set_title("Action-conditioned contact belief", loc="left", fontsize=12)
    contact_axis.legend(loc="lower right", frameon=False, fontsize=9)

    maximum_position = float(np.max(upper_x)) + 8.0
    scene_axis.set_xlim(-8.0, maximum_position)
    scene_axis.set_ylim(0.05, 1.0)
    scene_axis.set_yticks([0.25, 0.50, 0.75])
    scene_axis.set_yticklabels(["Ground truth", "Causal4D", "Persistence"])
    scene_axis.set_xlabel("Current material-point displacement (mm)")
    scene_axis.set_title("Current frame", loc="left", fontsize=12)
    scene_axis.grid(True, axis="x", color="#d9dde3", alpha=0.75)
    for y_value in (0.25, 0.50, 0.75):
        scene_axis.axhline(y_value, color="#eceff1", linewidth=1.0)
    scene_truth = scene_axis.plot([], [], "o", color="#202124", ms=13)[0]
    scene_dynamic = scene_axis.plot([], [], "o", color="#00897b", ms=13)[0]
    scene_static = scene_axis.plot([], [], "o", color="#c62828", ms=13)[0]
    status_text = scene_axis.text(
        0.04,
        0.94,
        "",
        transform=scene_axis.transAxes,
        va="top",
        fontsize=11,
        linespacing=1.45,
    )
    scene_axis.text(
        0.04,
        0.41,
        (
            "Aggregate result (40 cases)\n"
            f"{aggregate['mean_static_persistence_rmse_mm']:.2f} -> "
            f"{aggregate['mean_dynamic_contact_rmse_mm']:.3f} mm RMSE\n"
            f"{aggregate['mean_relative_rmse_improvement_percent']:.2f}% reduction\n"
            f"Onset error: {aggregate['mean_contact_onset_absolute_error_frames']:.3f} frames\n"
            f"Registered gates: {aggregate['gate_pass_count']}/{aggregate['case_count']}"
        ),
        transform=scene_axis.transAxes,
        va="top",
        fontsize=10.5,
        linespacing=1.55,
        color="#263238",
    )
    figure.text(
        0.012,
        0.014,
        "Interpretation boundary: mechanism evidence only; not real-object validation.",
        fontsize=9.5,
        color="#5f6368",
    )

    def update(display_frame: int) -> tuple[Any, ...]:
        frame = int(display_frame)
        truth_marker.set_data([frame], [truth_x[frame]])
        static_marker.set_data([frame], [static_x[frame]])
        dynamic_marker.set_data([frame], [dynamic_x[frame]])
        trajectory_cursor.set_xdata([frame, frame])
        contact_cursor.set_xdata([frame, frame])
        contact_marker.set_data([frame], [sticking_probability[frame]])
        scene_truth.set_data([truth_x[frame]], [0.25])
        scene_dynamic.set_data([dynamic_x[frame]], [0.50])
        scene_static.set_data([static_x[frame]], [0.75])
        phase = "observed prefix" if frame < prefix else "held-out future"
        contact_state = (
            "active" if trace.command_activation[frame] > 0.5 else "inactive"
        )
        status_text.set_text(
            f"Frame {frame:02d} / {len(frames) - 1:02d}\n"
            f"Phase: {phase}\n"
            f"Command: {contact_state}\n"
            f"P(sticking): {sticking_probability[frame]:.3f}"
        )
        return (
            truth_marker,
            static_marker,
            dynamic_marker,
            trajectory_cursor,
            contact_cursor,
            contact_marker,
            scene_truth,
            scene_dynamic,
            scene_static,
            status_text,
        )

    repeated_frames = np.repeat(frames, 2)
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=repeated_frames,
        interval=1000.0 / fps,
        blit=False,
    )
    movie.save(
        output_dir / "causal4d_dynamic_contact_demo.mp4",
        writer=animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=2200,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
        dpi=100,
    )
    movie.save(
        output_dir / "causal4d_dynamic_contact_demo.gif",
        writer=animation.PillowWriter(fps=fps),
        dpi=80,
    )
    update(len(frames) - 1)
    figure.savefig(
        output_dir / "causal4d_dynamic_contact_poster.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def render_demo(
    output_dir: Path,
    *,
    seed: int = 0,
    prefix_frame_count: int = 6,
    fps: int = 4,
) -> dict[str, Any]:
    """Render the video bundle and return its machine-readable summary."""

    if fps < 1:
        raise ValueError("fps must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_demo_cases()
    trace = delayed_contact_trace(
        seed=seed,
        prefix_frame_count=prefix_frame_count,
    )
    if trace.summary["future_observations_read"] != 0:
        raise RuntimeError("demo trace crossed the causal observation boundary")
    payload = {
        "schema_version": 1,
        "title": "Causal4D controlled delayed-contact demo",
        "claim_boundary": "controlled_simulation_only",
        "aggregate": aggregate,
        "representative_case": trace.summary,
        "artifacts": {
            "mp4": "causal4d_dynamic_contact_demo.mp4",
            "gif": "causal4d_dynamic_contact_demo.gif",
            "poster": "causal4d_dynamic_contact_poster.png",
        },
    }
    _render_animation(trace, aggregate, output_dir, fps=fps)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _summary_markdown(aggregate, trace.summary),
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix-frame-count", type=int, default=6)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--require-gates", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = render_demo(
        args.output_dir,
        seed=args.seed,
        prefix_frame_count=args.prefix_frame_count,
        fps=args.fps,
    )
    if args.require_gates and not payload["aggregate"]["all_gates_passed"]:
        raise SystemExit("controlled demo failed one or more registered gates")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
